# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file contains helper functions to generate reports for the
# executorch_delegation_runner.

from collections import defaultdict
from typing import Dict, List

from executorch_delegation_runner import ExceptionInfo, RunInfo


# Report generation functions
def _get_exception_str(e: ExceptionInfo):
    lines = []
    lines.append(f"type:{e.typ}")
    lines.append(f"msg:\n{e.msg}")
    lines.append(f"traceback:\n{e.tbk}")
    return "\n\n".join(lines)


def _get_line():
    return "-" * 80


def _get_successful_stage_report(infos: Dict[str, RunInfo]) -> str:
    lines = []
    lines.append(_get_line())
    lines.append("Successful stage completions")
    lines.append(_get_line())

    k = next(iter(infos))
    stages = list(infos[k].is_stage_successful.keys())
    lines.append(f"Starting: {len(infos)}")
    for stage in stages:
        lines.append(
            f"{stage}: {sum(infos[k].is_stage_successful[stage] for k in infos)}"
        )

    return "\n".join(lines)


def _get_error_per_stage_report(infos: Dict[str, RunInfo], stages: List[str]) -> str:
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

                exception_str = (
                    f"type:{infos[k].first_exception.typ}\n{tag}:{msg_or_tbk}"
                )
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

                lines.append(_get_line())
                lines.append(f"Unique errors at stage {stage}")
                lines.append(_get_line())
                for i, e in enumerate(error_to_repros):
                    lines.append(
                        f"\nError ({(i+1)}/{len(error_to_repros)}).  This repros on torchbench models {error_to_repros[e]}."
                    )
                    lines.append(e)

                is_first_error = False

    return "\n".join(lines)


def get_export_report(infos: Dict[str, RunInfo]) -> str:
    return _get_error_per_stage_report(infos, stages=["export"])


def get_delegation_report(infos: Dict[str, RunInfo]) -> str:
    lines = []
    lines.append(_get_successful_stage_report(infos))
    lines.append("\n")
    lines.append(
        _get_error_per_stage_report(
            infos, stages=["to_edge", "to_backend", "to_executorch", "to_buffer"]
        )
    )
    return "\n".join(lines)


def get_exception_reports(infos: Dict[str, RunInfo]) -> Dict[str, str]:
    res = {}
    for model_name, info in infos.items():
        if info.first_exception is not None:
            res[model_name] = _get_exception_str(info.first_exception)
    return res
