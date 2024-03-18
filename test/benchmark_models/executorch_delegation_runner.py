# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
logger = logging.getLogger()

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.quantizer import Quantizer
from executorch.exir import to_edge, EdgeCompileConfig, EdgeProgramManager, ExecutorchProgramManager, ExecutorchBackendConfig
from typing import Optional
from torch.export import export, ExportedProgram

from executorch.exir.backend.backend_api import to_backend, Partitioner
from executorch.exir.backend.backend_details import CompileSpec
import torch
import copy
from dataclasses import dataclass, fields
from functools import wraps
import time
import torch.nn as nn
from typing import List, Dict
import traceback

# Timeout wrapper copied from here:
# https://baites.github.io/computer-science/patterns/2018/05/14/timeouts-in-python-and-why-you-should-use-python3.html
from concurrent.futures import ThreadPoolExecutor
def ftimeout(timeout):
    """Implement function timeout decorator."""
    def decorator(callback):
        @wraps(callback)
        def wrapper(*args, **kargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(callback, *args, **kargs)
                return future.result(timeout=timeout)
        return wrapper
    return decorator

# Define a global timeout wrapper for all stages.
_TIMEOUT_WRAPPER = ftimeout(60 * 20) # 20 minutes



@dataclass
class ToBackendSpec:
    partitioner: Partitioner
    quantizer: Optional[Quantizer]

def get_to_backend_spec(delegation: str):
    if delegation == "no_backend":
        return ToBackendSpec(partitioner=None, quantizer=None)
    elif delegation == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        return ToBackendSpec(partitioner=XnnpackPartitioner(), quantizer=None)
    elif delegation == "xnnpack_quantized":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        return ToBackendSpec(partitioner=XnnpackPartitioner(), quantizer=quantizer)
    elif delegation == "mps":
        from executorch.backends.apple.mps.mps_preprocess import MPSBackend
        from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner
        compile_specs = [CompileSpec("use_fp16", bytes([1]))]
        partitioner = MPSPartitioner(compile_specs=compile_specs)
        return ToBackendSpec(partitioner=partitioner, quantizer=None)
    elif delegation == "coreml":
        # although not "used" below, CoreMLBackend does need to be imported or we will
        # get a backend not found errror during partitioning.  This should probably be
        # importred during __init__.py in executorch.backends.apple.coreml.partition.coreml_partitioner
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
        return ToBackendSpec(partitioner=CoreMLPartitioner(), quantizer=None)
    else:
        raise NotImplementedError(f"Unknown delegation {delegation}.")

@dataclass
class _StageOutputs:
    eval_inputs: Optional[object] = None
    quantize: Optional[torch.nn.Module] = None
    export: Optional[ExportedProgram] = None
    to_edge: Optional[EdgeProgramManager] = None
    to_backend: Optional[EdgeProgramManager] = None
    to_executorch: Optional[ExecutorchProgramManager] = None
    buffer: Optional[bytes] = None

@dataclass
class _StageInfo:
    runtime_s: float
    exception: Optional[Exception] = None

    @property
    def success(self) -> bool:
        return self.exception is None

@dataclass
class _StageInfos:
    eval_inputs: Optional[_StageInfo] = None
    quantize: Optional[_StageInfo] = None
    export: Optional[_StageInfo] = None
    to_edge: Optional[_StageInfo] = None
    to_backend: Optional[_StageInfo] = None
    to_executorch: Optional[_StageInfo] = None
    buffer: Optional[_StageInfo] = None

def _wrap_stage(stage_method):
    @wraps(stage_method)
    def wrapper(*args, **kwargs):
        self_obj = args[0]
        stage_name = stage_method.__name__

        print(f"Running {stage_name}...", end="")
        start_time = time.perf_counter()
        exception = None
        try:
            stage_method(*args, **kwargs)
        except Exception as e:
            exception = e
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Record stage info
        setattr(self_obj.stage_infos, stage_name, _StageInfo(
            runtime_s=elapsed_time,
            exception=exception,
        ))

        if exception is not None:
            # Set stage_output to None
            setattr(self_obj.stage_outputs, stage_name, None)

        status = "SUCCEEDED" if exception is None else "FAILED"
        print(f"{status} after {elapsed_time:.6f} seconds.")

    return wrapper

@dataclass
class ExceptionInfo:
    typ: str
    msg: str
    tbk: str

    def __init__(self, e: Exception):
        self.typ = str(type(e))
        self.msg = str(e)
        lines = traceback.format_exception(type(e), e, e.__traceback__)
        self.tbk = ''.join(lines)

@dataclass
class RunInfo:
    is_stage_successful: Dict[str, bool]
    stage_runtime_s: Dict[str, float]

    first_failed_stage: Optional[str]
    first_exception: Optional[ExceptionInfo]

class ExecuTorchDelegationRunner:
    def __init__(self, model: nn.Module, example_inputs, to_backend_spec: ToBackendSpec) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.partitioner = to_backend_spec.partitioner
        self.quantizer = to_backend_spec.quantizer
        self.stage_outputs = _StageOutputs()
        self.stage_infos = _StageInfos()
        self.is_run = False

    @_wrap_stage
    @_TIMEOUT_WRAPPER
    def eval_inputs(self):
        assert self.model, "self.model does not exist"
        assert self.example_inputs, "self.example_inputs does not exist"
        self.stage_outputs.eval_inputs = self.model(*self.example_inputs)

    @_wrap_stage
    @_TIMEOUT_WRAPPER
    def quantize(self):
        assert self.quantizer, "self.quantizer does not exist"
        assert self.stage_outputs.eval_inputs is not None, "Dependency does not exist"

        # Quantizer must be applied to pre_autograd_graph
        pre_autograd_graph = capture_pre_autograd_graph(self.model, self.example_inputs)

        # Prepare for calibration
        prepared_model = prepare_pt2e(pre_autograd_graph, self.quantizer)

        # Calibrate (suppresses warning about quant observer not seeing data)
        prepared_model(*self.example_inputs)

        # Quantize
        self.stage_outputs.quantize = convert_pt2e(prepared_model)

    @_wrap_stage
    @_TIMEOUT_WRAPPER
    def export(self):
        if self.quantizer:
            model_to_export = self.stage_outputs.quantize
        else:
            assert self.stage_outputs.eval_inputs is not None, "Dependency does not exist"
            model_to_export = self.model

        assert model_to_export, "Dependency does not exist"

        self.stage_outputs.export = export(model_to_export, self.example_inputs)

    @_wrap_stage
    @_TIMEOUT_WRAPPER
    def to_edge(self):
        assert self.stage_outputs.export, "Dependency does not exist"
        self.stage_outputs.to_edge = to_edge(self.stage_outputs.export, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    @_wrap_stage
    @_TIMEOUT_WRAPPER
    def to_backend(self):
        assert self.partitioner, "self.partitioner does not exist"
        assert self.stage_outputs.to_edge, "Dependency does not exist"
        self.stage_outputs.to_backend = self.stage_outputs.to_edge.to_backend(self.partitioner)

    @_wrap_stage
    @_TIMEOUT_WRAPPER
    def to_executorch(self):
        if self.partitioner:
            model_to_executorch = self.stage_outputs.to_backend
        else:
            model_to_executorch = self.stage_outputs.to_edge
        assert model_to_executorch, "Dependency does not exist"

        # to_executorch modifies the EdgeProgramManager, so we make a copy before
        # calling it to help with debugging.
        model_to_executorch = copy.deepcopy(model_to_executorch)
        self.stage_outputs.to_executorch = model_to_executorch.to_executorch(config=ExecutorchBackendConfig(extract_constant_segment=False))

    @_wrap_stage
    @_TIMEOUT_WRAPPER
    def buffer(self):
        assert self.stage_outputs.to_executorch, "Dependency does not exist"
        self.stage_outputs.buffer = self.stage_outputs.to_executorch.buffer

    def stages(self) -> List[str]:
        stages = ["eval_inputs"]
        if self.quantizer:
            stages.append("quantize")
        stages.append("export")
        stages.append("to_edge")
        if self.partitioner:
            stages.append("to_backend")
        stages.append("to_executorch")
        stages.append("buffer")
        return stages

    def run(self):
        self.model.eval()
        for stage in self.stages():
            getattr(self, stage)()

        self.is_run = True

    def run_info(self) -> RunInfo:
        assert self.is_run

        is_stage_successful = {
            stage: getattr(getattr(self.stage_infos, stage), "success")
            for stage in self.stages()
        }

        stage_runtime_s = {
            stage: getattr(getattr(self.stage_infos, stage), "runtime_s")
            for stage in self.stages()
        }

        first_failed_stage = None
        first_exception = None
        for stage in self.stages():
            if not getattr(getattr(self.stage_infos, stage), "success"):
                first_failed_stage = stage
                first_exception = ExceptionInfo(getattr(getattr(self.stage_infos, stage), "exception"))
                break

        return RunInfo(
            is_stage_successful,
            stage_runtime_s,
            first_failed_stage,
            first_exception
        )
