# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import os
import pickle
import pkgutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.bundled_inputs
from executorch.exir import ExecutorchProgramManager
from executorch_delegation_runner import (
    ExceptionInfo,
    ExecuTorchDelegationRunner,
    get_to_backend_spec,
    RunInfo,
)
from report_generation import (
    get_delegation_report,
    get_exception_reports,
    get_export_report,
)
from torch.export import ExportedProgram
from torch.utils.bundled_inputs import bundle_large_tensor
from torch.utils.mobile_optimizer import optimize_for_mobile


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
        benchmark = benchmark_cls(test="eval", device="cpu")
        model, example = benchmark.get_module()
        model.eval()
        return model, example


class ReaderWriter:
    def __init__(self, directory: str, delegation: str, directory_exist_ok: bool):
        self.directory = os.path.abspath(directory)
        os.makedirs(directory, exist_ok=directory_exist_ok)
        self.delegation = delegation  # delegation will prefix all generated files

        # Create subdirectories

        # Trackers are empty files used to skip successful runs
        # To re-run a specific model, you need to delete its tracker file
        os.makedirs(f"{directory}/trackers", exist_ok=True)
        self.trackers_directory = f"{directory}/trackers"

        os.makedirs(f"{directory}/reports", exist_ok=True)
        self.reports_directory = f"{directory}/reports"

        os.makedirs(f"{directory}/exceptions", exist_ok=True)
        self.exceptions_directory = f"{directory}/exceptions"

        os.makedirs(f"{directory}/models", exist_ok=True)
        self.models_directory = f"{directory}/models"

        # run_infos contains pickled RunInfo objects.  The information in RunInfo
        # is already parsed and put in the other directories as plain txt.
        os.makedirs(f"{directory}/run_infos", exist_ok=True)
        self.run_infos_directory = f"{directory}/run_infos"

    def write_report(self, name: str, report: str):
        path = f"{self.reports_directory}/{self.delegation}_{name}_report.txt"
        with open(path, "w") as f:
            f.write(report)

    def write_run_info(self, name: str, run_info: RunInfo):
        path = f"{self.run_infos_directory}/{self.delegation}_{name}_run_info.pkl"
        with open(path, "wb") as f:
            pickle.dump(run_info, f)

    def write_et_model(
        self, name: str, executorch_program_manager: ExecutorchProgramManager
    ):
        path = f"{self.models_directory}/{self.delegation}_{name}_model.pte"
        with open(path, "wb") as f:
            executorch_program_manager.write_to_file(f)

    def write_lite_model(self, name: str, bundled_opt_model):
        bundled_opt_model._save_for_lite_interpreter(
            f"{self.models_directory}/{self.delegation}_{name}_model.ptl"
        )

    def write_example_inputs(self, name: str, example_inputs):
        path = f"{self.models_directory}/{self.delegation}_{name}_ex_inputs.pt"
        with open(path, "wb") as f:
            torch.save(example_inputs, f)

    def write_eval_example_inputs(self, name: str, example_inputs):
        path = f"{self.models_directory}/{self.delegation}_{name}_eval_ex_inputs.pt"
        with open(path, "wb") as f:
            torch.save(example_inputs, f)

    def write_exception(self, name: str, exception: str):
        path = (
            f"{self.exceptions_directory}/{self.delegation}_{name}_first_exception.txt"
        )
        with open(path, "w") as f:
            f.write(exception)

    def write_tracker(self, name: str):
        path = f"{self.trackers_directory}/{self.delegation}_{name}.txt"
        with open(path, "w") as f:
            f.write("")

    def tracker_exists(self, name: str):
        path = f"{self.trackers_directory}/{self.delegation}_{name}.txt"
        return os.path.exists(path)

    def read_run_info(self, name: str) -> RunInfo:
        path = f"{self.run_infos_directory}/{self.delegation}_{name}_run_info.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def read_run_infos(self, model_names: List[str]) -> Dict[str, RunInfo]:
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
            if reader_writer.tracker_exists(model_name):
                print(
                    f"Skipping {model_name} because it already ran on {delegation}.\nTo rerun, delete the tracker file or set skip_existing=False."
                )
                continue

        extractable = False
        runner = None
        try:
            model, example_inputs = model_loader.load_model_and_example_inputs(
                model_name
            )
            extractable = True
        except Exception as e:
            print("Extraction exception: ", str(e)[0:50])
            non_extractable.append(model_name)

        # runner should not throw
        if extractable:
            runner = ExecuTorchDelegationRunner(
                model=model,
                example_inputs=example_inputs,
                to_backend_spec=get_to_backend_spec(delegation),
            )
            runner.run()
            reader_writer.write_run_info(model_name, runner.run_info())
            if runner.stage_outputs.buffer is not None:
                reader_writer.write_et_model(
                    model_name, runner.stage_outputs.to_executorch
                )
                reader_writer.write_example_inputs(model_name, runner.example_inputs)
                reader_writer.write_eval_example_inputs(
                    model_name, runner.stage_outputs.eval_inputs
                )

        # Write tracker
        reader_writer.write_tracker(model_name)

    print(
        f"\nThe following {len(non_extractable)} models could not be extracted: {non_extractable}"
    )

    # Generate reports
    run_infos = {}
    for model_name in model_names:
        if reader_writer.tracker_exists(model_name):
            run_infos[model_name] = reader_writer.read_run_info(model_name)

    if len(run_infos) > 0:
        reader_writer.write_report("export", get_export_report(run_infos))
        reader_writer.write_report("summary", get_delegation_report(run_infos))
        exception_reports = get_exception_reports(run_infos)
        for model_name, report in exception_reports.items():
            reader_writer.write_exception(model_name, report)


# Runs export for the lite interpreter
# Unlike the ET flow, we do not collect errors.
def run_lite(model_loader, model_names, reader_writer, skip_existing):
    non_extractable = []
    for i, model_name in enumerate(model_names):
        print(f"\n{model_name} ({i+1}/{len(model_names)})")
        if skip_existing:
            if reader_writer.tracker_exists(model_name):
                print(
                    f"Skipping {model_name} because it already ran on {delegation}.\nTo rerun, delete the tracker file or set skip_existing=False."
                )
                continue

        extractable = False
        try:
            model, example_inputs = model_loader.load_model_and_example_inputs(
                model_name
            )
            extractable = True
        except Exception as e:
            print("Extraction exception: ", str(e)[0:50])
            non_extractable.append(model_name)
            continue

        # Export to lite
        try:
            traced_script_module = torch.jit.trace(model, example_inputs)
            opt_model = optimize_for_mobile(traced_script_module)
            bundled_opt_model = torch.utils.bundled_inputs.bundle_inputs(
                opt_model,
                {opt_model.forward: [(bundle_large_tensor(example_inputs[0]),)]},
            )
            reader_writer.write_lite_model(model_name, bundled_opt_model)
        except Exception as e:
            print(f"Exception: {str(e)[0:50]}")

        # Write tracker
        reader_writer.write_tracker(model_name)

    print(
        f"\nThe following {len(non_extractable)} models could not be extracted: {non_extractable}"
    )


# Torchbench models to exclude from AOT flow
AOT_EXCLUSIONS: List[str] = [
    # The following models do not load from torchbench because they
    # require CUDA or some other dependency not available on Mac out-of-the-box
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
    # These models use too much memory
    "moondream",
    "demucs",
    "timm_vision_transformer_large",
]

# Subset of torchbench models that are appropriate to run on the limited
# resources of mobile devices.
# For AOT flow, we can still test exportability of non-mobile models because
# the issues found are still valuable in general
MOBILE_MODELS = [
    "hf_DistilBert",
    "basic_gnn_sage",
    "pytorch_unet",
    "lennard_jones",
    "phlippe_resnet",
    "mobilenet_v2",
    "dcgan",
    "functorch_dp_cifar10",
    "squeezenet1_1",
    "timm_efficientnet",
    "maml_omniglot",
    "resnet18",
    "mnasnet1_0",
    "resnet50",
    "LearningToPaint",
    "shufflenet_v2_x1_0",
    "mobilenet_v3_large",
    "resnext50_32x4d",
]


if __name__ == "__main__":
    import argparse
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    logging.disable(logging.CRITICAL)

    VALID_DELEGATIONS: List[str] = [
        "no_backend",
        "xnnpack",
        "xnnpack_quantized",
        "mps",
        "coreml",
        "lite",
    ]

    parser = argparse.ArgumentParser(
        description="Run torchbench models through the ExecuTorch AOT and lite interpreter flows.  This must be run in the same directory as the `benchmark` folder after installing torchbench."
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default="./out",
        help="Directory to store outputs.  Default is ./out.",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        default=False,
        help="Whether it is OK if the write_dir already exists.  If on, data can be overwritten.",
    )
    parser.add_argument(
        "--mobile_models_only",
        action="store_true",
        default=False,
        help="Whether to only delegate models that are appropriate to run in a mobile environment.  To catch more AOT errors, do not set this flag.",
    )
    parser.add_argument(
        "--delegations",
        nargs="+",
        help=f"How to delegate the models.  Available options are: {VALID_DELEGATIONS}",
    )
    args = parser.parse_args()

    for d in args.delegations:
        assert d in VALID_DELEGATIONS

    model_loader = TorchbenchModelLoader()
    if args.mobile_models_only:
        model_names = MOBILE_MODELS
    else:
        model_names = [
            n for n in model_loader.get_model_names() if n not in AOT_EXCLUSIONS
        ]

    for delegation in args.delegations:
        print(f"RUNNING {delegation}")
        reader_writer = ReaderWriter(args.write_dir, delegation, args.exist_ok)

        if delegation == "lite":
            run_lite(
                model_loader=model_loader,
                model_names=model_names,
                reader_writer=reader_writer,
                skip_existing=True,
            )
        else:
            # Run ET flow
            run(
                model_loader=model_loader,
                delegation=delegation,
                model_names=model_names,
                reader_writer=reader_writer,
                skip_existing=True,
            )
