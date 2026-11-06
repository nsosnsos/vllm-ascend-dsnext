#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Whether to enable the experimental DeepSeek-V4 private circle pool.
    # Valid values: 0 (default, original pool/prefix-cache flow) or 1
    # (compact private pool plus prefix-hit bounded replay by configured SWA window).
    # Non-sensitive; read on scheduler/runner initialization and hot paths.
    "VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL": lambda: bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL", "0"))
    ),
    # Whether to enable the experimental DeepSeek-V4.1 tail-window prefill:
    # layers after the last KV source compute only each request's trailing
    # SWA-window rows. Valid values are 0 (default, full-chunk compute) and 1.
    # Requires VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL=1; silently stays inert
    # on steps without a private-circle prefill plan. Non-sensitive; read on
    # the model forward and attention hot paths.
    "VLLM_ASCEND_ENABLE_TAIL_WINDOW_PREFILL": lambda: bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_TAIL_WINDOW_PREFILL", "0"))
    ),
    # Whether to run the DeepSeek-V4.1 indexer query/weight prologue through the
    # fused ``indexer_prologue_qw`` operator (MXFP8 Q projection + partial RoPE +
    # MXFP4 Q quantization, plus the BF16 head-weight projection). Valid values
    # are 0 (default, the eager projection chain) and 1 (fused operator).
    # Enabling it also makes the DS linear scheme keep ``indexer.wq_b``
    # output-major for the operator. Requires the A5 net-ops wheel providing
    # ``indexer_prologue_qw``. Non-sensitive.
    "VLLM_ASCEND_ENABLE_INDEXER_PROLOGUE_QW": lambda: bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_INDEXER_PROLOGUE_QW", "0"))
    ),
    # Whether to enable lightweight request timeline tracing. Valid values are
    # 0 (default, disabled) and 1 (enabled). This value is non-sensitive and is
    # read during platform and worker patch initialization.
    "VLLM_ASCEND_TRACE": lambda: bool(int(os.getenv("VLLM_ASCEND_TRACE", "0"))),
    # Role attached to trace records. Expected values are "prefill" or
    # "decode"; the default identifies an unspecified role. Non-sensitive.
    "ROLE": lambda: os.getenv("ROLE", "unknown_role"),
    # Directory for per-process trace logs. Non-sensitive and read when the
    # trace utility module is initialized.
    "TRACE_OUTPUT_DIRECTORY": lambda: os.getenv("TRACE_OUTPUT_DIRECTORY", "/tmp/trace_output_directory"),
    # Optional custom trace namelist path. Non-sensitive and read when trace
    # patches are initialized.
    "PROFILING_NAMELIST": lambda: os.getenv("PROFILING_NAMELIST", None),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
