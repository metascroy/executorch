# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from .meta_registrations import *  # noqa

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from ...portable.utils import export_to_edge, save_pte_program

from .quantizer import (
    QuantFusion,
    ReplacePT2DequantWithXtensaDequant,
    ReplacePT2QuantWithXtensaQuant,
    XtensaQuantizer,
)


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def export_xtensa_model(model, example_inputs):
    # Quantizer
    quantizer = XtensaQuantizer()

    # Export
    model_exp = capture_pre_autograd_graph(model, example_inputs)

    # Prepare
    prepared_model = prepare_pt2e(model_exp, quantizer)
    prepared_model(*example_inputs)

    # Convert
    converted_model = convert_pt2e(prepared_model)

    # pyre-fixme[16]: Pyre doesn't get that XtensaQuantizer has a patterns attribute
    patterns = [q.pattern for q in quantizer.quantizers]
    QuantFusion(patterns)(converted_model)

    # pre-autograd export. eventually this will become torch.export
    converted_model_exp = capture_pre_autograd_graph(converted_model, example_inputs)

    converted_model_exp = torch.ao.quantization.move_exported_model_to_eval(
        converted_model_exp
    )

    exec_prog = (
        export_to_edge(
            converted_model_exp,
            example_inputs,
            edge_compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )
        .transform(
            [ReplacePT2QuantWithXtensaQuant(), ReplacePT2DequantWithXtensaDequant()]
        )
        .to_executorch(config=ExecutorchBackendConfig())
    )

    logging.info(f"Final exported graph:\n{exec_prog.exported_program().graph}")

    # Save the program as XtensaDemoModel.pte
    save_pte_program(exec_prog.buffer, "XtensaDemoModel")
