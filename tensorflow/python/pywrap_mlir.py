# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Python module for MLIR functions exported by pybind11."""

from typing import List, Optional

from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *

def _encode_str(value: Optional[str]) -> Optional[bytes]:
    return value.encode('utf-8') if value is not None else None

def import_graphdef(
    graphdef: str,
    pass_pipeline: str,
    show_debug_info: bool,
    input_names: Optional[List[str]] = None,
    input_data_types: Optional[List[str]] = None,
    input_data_shapes: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
) -> bytes:
    if input_names is not None:
        return ImportGraphDef(
            _encode_str(graphdef),
            _encode_str(pass_pipeline),
            show_debug_info,
            _encode_str(','.join(input_names)),
            _encode_str(','.join(input_data_types)),
            _encode_str(':'.join(input_data_shapes)),
            _encode_str(','.join(output_names or [])),
        )
    return ImportGraphDef(
        _encode_str(graphdef),
        _encode_str(pass_pipeline),
        show_debug_info,
    )

def import_function(
    concrete_function: any,
    pass_pipeline: str,
    show_debug_info: bool,
) -> bytes:
    ctxt = context.context()
    ctxt.ensure_initialized()
    return ImportFunction(
        ctxt._handle,
        _encode_str(str(concrete_function.function_def)),
        _encode_str(pass_pipeline),
        show_debug_info,
    )

def experimental_convert_saved_model_to_mlir(
    saved_model_path: str,
    exported_names: str,
    show_debug_info: bool,
) -> bytes:
    return ExperimentalConvertSavedModelToMlir(
        _encode_str(saved_model_path),
        _encode_str(exported_names),
        show_debug_info,
    )

def experimental_convert_saved_model_v1_to_mlir_lite(
    saved_model_path: str,
    exported_names: str,
    tags: str,
    upgrade_legacy: bool,
    show_debug_info: bool,
) -> bytes:
    return ExperimentalConvertSavedModelV1ToMlirLite(
        _encode_str(saved_model_path),
        _encode_str(exported_names),
        _encode_str(tags),
        upgrade_legacy,
        show_debug_info,
    )

def experimental_convert_saved_model_v1_to_mlir(
    saved_model_path: str,
    exported_names: str,
    tags: str,
    lift_variables: bool,
    include_variables_in_initializers: bool,
    upgrade_legacy: bool,
    show_debug_info: bool,
) -> bytes:
    return ExperimentalConvertSavedModelV1ToMlir(
        _encode_str(saved_model_path),
        _encode_str(exported_names),
        _encode_str(tags),
        lift_variables,
        include_variables_in_initializers,
        upgrade_legacy,
        show_debug_info,
    )

def experimental_run_pass_pipeline(
    mlir_txt: str,
    pass_pipeline: str,
    show_debug_info: bool,
) -> bytes:
    return ExperimentalRunPassPipeline(
        _encode_str(mlir_txt), _encode_str(pass_pipeline), show_debug_info
    )

def experimental_write_bytecode(
    filename: str,
    mlir_txt: str,
) -> bytes:
    return ExperimentalWriteBytecode(
        _encode_str(filename), _encode_str(mlir_txt)
    )

def experimental_tflite_to_tosa_bytecode(
    flatbuffer: str,
    bytecode: str,
    use_external_constant: bool = False,
    ordered_input_arrays: Optional[List[str]] = None,
    ordered_output_arrays: Optional[List[str]] = None,
) -> bytes:
    if ordered_input_arrays is None:
        ordered_input_arrays = []
    if ordered_output_arrays is None:
        ordered_output_arrays = []
    return ExperimentalTFLiteToTosaBytecode(
        _encode_str(flatbuffer),
        _encode_str(bytecode),
        use_external_constant,
        ordered_input_arrays,
        ordered_output_arrays,
    )
