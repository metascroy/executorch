# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch_delegation_runner import (
    ExecuTorchDelegationRunner,
    get_to_backend_spec,
    ExceptionInfo,
    RunInfo,
)
import sys
import pkgutil
import importlib
import os
import pickle
import torch
from torch.export import ExportedProgram
from typing import Optional, Tuple, List, Dict
import logging
from collections import defaultdict
from executorch.exir import ExecutorchProgramManager

# Set up torchbench model loading
# This script must be in the same directory as the benchmark folder
class TorchbenchModelLoader:
    def __init__(self):
        sys.path.append("benchmark")
        import torchbenchmark.models
        submodules = list(pkgutil.iter_modules(torchbenchmark.models.__path__))
        model_names = []
        for s in submodules:
            model_names.append(s.name)
        self.model_names = sorted(model_names)

    def get_model_names(self):
        return self.model_names

    def load_model_and_example_inputs(self, model_name):
        assert model_name in self.model_names

        model_name = f"torchbenchmark.models.{model_name}"
        module = importlib.import_module(model_name)

        benchmark_cls = getattr(module, "Model", None)
        benchmark = benchmark_cls(test="eval", device = "cpu")
        model, example = benchmark.get_module()
        model.eval()
        return model, example

class ReaderWriter:
    def __init__(self, directory, delegation):
        self.directory = directory
        #os.makedirs(directory, exist_ok=False)
        self.delegation = delegation

    def write_delegation_report(self, report: str):
        path = f"{self.directory}/{self.delegation}_report.txt"
        with open(path, "w") as f:
            f.write(report)

    def write_export_report(self, report: str):
        path = f"{self.directory}/{self.delegation}_export_report.txt"
        with open(path, "w") as f:
            f.write(report)

    def write_run_info(self, run_info: RunInfo, model_name: str):
        path = f"{self.directory}/{self.delegation}_{model_name}_run_info.pkl"
        with open(path, "wb") as f:
            pickle.dump(run_info, f)

    def write_pte(self, executorch_program_manager: ExecutorchProgramManager, model_name: str):
        path = f"{self.directory}/{self.delegation}_{model_name}_model.pte"
        with open(path, "wb") as f:
            executorch_program_manager.write_to_file(f)

    def write_example_inputs(self, example_inputs, model_name: str):
        path = f"{self.directory}/{self.delegation}_{model_name}_ex_inputs.pt"
        with open(path, "wb") as f:
            torch.save(example_inputs, f)

    def write_eval_example_inputs(self, example_inputs, model_name: str):
        path = f"{self.directory}/{self.delegation}_{model_name}_eval_ex_inputs.pt"
        with open(path, "wb") as f:
            torch.save(example_inputs, f)

    def write_exception(self, first_exception_str: str, model_name: str):
        path = f"{self.directory}/{self.delegation}_{model_name}_first_exception.txt"
        with open(path, "w") as f:
            f.write(first_exception_str)

    def read_run_info(self, model_name, no_raise=False) -> Optional[RunInfo]:
        path = f"{self.directory}/{self.delegation}_{model_name}_run_info.pkl"
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            if no_raise:
                logging.warning(f"Could not load {model_name} because the file {path} does not exist.")
                return None
            else:
                raise e

    def read_run_infos(self,  model_names: List[str]) -> Dict[str, RunInfo]:
        run_infos = {}
        for model_name in model_names:
            run_info = self.read_run_info(model_name, no_raise=True)
            if run_info is not None:
                run_infos[model_name] = run_info
        return run_infos

def run(model_loader, delegation, model_names, reader_writer, skip_existing):
    non_extractable = []
    for i, model_name in enumerate(model_names):
        print(f"\n{model_name} ({i+1}/{len(model_names)})")
        if skip_existing:
            r = reader_writer.read_run_info(model_name, no_raise=True)
            if r is not None:
                print(f"Skipping {model_name} because it already ran on {delegation}.\nTo rerun, delete the info file.")
                continue

        extractable = False
        runner = None
        try:
            model, example_inputs = model_loader.load_model_and_example_inputs(model_name)
            extractable = True
        except Exception as e:
            print("Extraction exception: ", str(e)[0:50])
            non_extractable.append(model_name)

        # runner should not throw
        if extractable:
            runner = ExecuTorchDelegationRunner(model=model, example_inputs=example_inputs, to_backend_spec=get_to_backend_spec(delegation))
            runner.run()

            reader_writer.write_run_info(runner.run_info(), model_name)
            if runner.stage_outputs.buffer is not None:
                reader_writer.write_pte(runner.stage_outputs.to_executorch, model_name)
                reader_writer.write_example_inputs(runner.example_inputs, model_name)
                reader_writer.write_eval_example_inputs(runner.stage_outputs.eval_inputs, model_name)

    print(f"\nThe following {len(non_extractable)} models could not be extracted: {non_extractable}")

# Report generation functions
def get_exception_str(e: ExceptionInfo):
    lines = []
    lines.append(f"type:{e.typ}")
    lines.append(f"msg:\n{e.msg}")
    lines.append(f"traceback:\n{e.tbk}")
    return "\n\n".join(lines)

def get_line():
    return "-"*100

def get_successful_stage_report(infos: Dict[str, RunInfo]) -> str:
    lines = []
    lines.append(get_line())
    lines.append("Successful stage completions")
    lines.append(get_line())

    k = next(iter(infos))
    stages = list(infos[k].is_stage_successful.keys())
    lines.append(f"Starting: {len(infos)}")
    for stage in stages:
        lines.append(f"{stage}: {sum(infos[k].is_stage_successful[stage] for k in infos)}")

    return "\n".join(lines)

def get_error_per_stage_report(infos: Dict[str, RunInfo], stages: List[str]) -> str:
    def collect_errors_at_stage(infos, stage):
        error_to_repros: Dict[str, List[str]] = defaultdict(list)
        for k in infos:
            if infos[k].first_failed_stage == stage:
                # Collect first chars of msg or last chars of traceback
                error_window = 300
                tag = "msg"
                msg_or_tbk = infos[k].first_exception.msg[0:error_window]
                if msg_or_tbk == "":
                    tag = "tbk"
                    msg_or_tbk = infos[k].first_exception.tbk[-error_window:-1]


                exception_str = f"type:{infos[k].first_exception.typ}\n{tag}:{msg_or_tbk}"
                error_to_repros[exception_str].append(k)
        return dict(error_to_repros)

    k = next(iter(infos))
    _stages = list(infos[k].is_stage_successful.keys())
    lines = []
    is_first_error = True
    for stage in _stages:
        if stage in stages:
            error_to_repros = collect_errors_at_stage(infos, stage)
            if len(error_to_repros) > 0:
                if not is_first_error:
                    lines.append("\n")

                lines.append(get_line())
                lines.append(f"Unique errors at stage {stage}")
                lines.append(get_line())
                for i, e in enumerate(error_to_repros):
                    lines.append(f"\nError ({(i+1)}/{len(error_to_repros)}).  This repros on torchbench models {error_to_repros[e]}.")
                    lines.append(e)

                is_first_error = False

    return "\n".join(lines)


if __name__ == "__main__":
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    logging.disable(logging.CRITICAL)

    model_loader = TorchbenchModelLoader()

    model_names = [n for n in model_loader.get_model_names() if n not in [
        # Exclusion list

        # Skipping the following because they do not load from
        # torchbench on my machine
        # (e.g., they require CUDA or I'm missing a dependency)
        "DALLE2_pytorch",
        "doctr_det_predictor",
        "doctr_reco_predictor",
        "llama_v2_7b_16h",
        "mobilenet_v2_quantized_qat",
        "moco",
        "resnet50_quantized_qat",
        "sam",
        "sam_fast",
        "simple_gpt",
        "simple_gpt_tp_manual",
        "stable_diffusion_text_encoder",
        "stable_diffusion_unet",
        "tacotron2",
        "timm_efficientdet",
        "torch_multimodal_clip",

        # Skipping the following because they use too much memory
        "moondream",
        "demucs",
        "timm_vision_transformer_large",
    ]]

    ###################################################
    # TODO: SET THESE CORRECTLY
    delegations = ["no_backend", "xnnpack", "xnnpack_quantized", "mps", "coreml"]
    write_dir = f"/Users/scroy/etorch/runs/test"
    ###################################################

    for delegation in delegations:
        reader_writer = ReaderWriter(write_dir, delegation)

        print(f"RUNNING {delegation}")

        run(model_loader=model_loader, delegation=delegation, model_names=model_names, reader_writer=reader_writer, skip_existing=True)

        # Generate reports
        infos = reader_writer.read_run_infos(model_names)

        # Export errors are common across all delegations, so we separate it out into its own report
        reader_writer.write_export_report(get_error_per_stage_report(infos, stages=["export"]))

        # Write delegation report
        lines = []
        lines.append(get_successful_stage_report(infos))
        lines.append("\n")
        lines.append(get_error_per_stage_report(infos, stages=["to_edge", "to_backend", "to_executorch", "to_buffer"]))
        reader_writer.write_delegation_report("\n".join(lines))

        # Generate detailed exception files
        for model_name, info in infos.items():
            if info.first_exception is not None:
                reader_writer.write_exception(get_exception_str(info.first_exception), model_name)
