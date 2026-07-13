# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__all__ = ["scope"]

from triton.language.core import _constexpr_to_value

_VALID_CORE_MODES = ("cube", "vector")
_VALID_VECTOR_MODES = ("simd", "simt")


class scope:
    """
    Context manager for entering and exiting a scope, where operations within a scope shares some common characteristics.

    Example:
    ```python
        import triton.language.extra.cann.extension as extension

        @triton.jit
        def kernel(x_ptr, y_ptr, N):
            # specify annotation
            with extension.scope(feature_a=True):
                a = tl.load(x_ptr)
                b = tl.load(y_ptr)
                result = tl.dot(a, b)
    ```

    Reserved keywords:
        - `core_mode`: Allows explicitly specify which core type should be used for operations within a code block, helping the compiler generate appropriate code for cube or vector cores.
        - `vector_mode`: Selects the vector-core execution mode for operations in the scope.
            Valid values: "simd", "simt". This is intended for mixed SIMD/SIMT
            lowering. The generated `vec_mode` scope attribute is consumed by
            TritonToUnstructure to request the SIMT indirect fast path when
            available.
    """

    def __init__(self, core_mode: str = None, _builder=None, _semantic=None,
                 vector_mode: str = None, vec_mode: str = None, **kwargs):
        """
        :param core_mode: Either "cube" or "vector" to specify the core type
        :param vector_mode: Either "simd" or "simt" to select the vector path
        :param vec_mode: Backward-compatible alias for vector_mode
        :param _builder: Internal builder object (set by code_generator)
        :param _semantic: Internal semantic object (set by code_generator)
        :param kwargs: Additional internal parameters
        """
        if vector_mode is not None and vec_mode is not None and vector_mode != vec_mode:
            raise ValueError("vector_mode and vec_mode cannot specify different values")

        raw_vector_mode = vector_mode if vector_mode is not None else vec_mode

        # Convert constexpr to value if not being called from code generator
        self.core_mode = _constexpr_to_value(core_mode) if _builder is None else core_mode
        self.vector_mode = (
            _constexpr_to_value(raw_vector_mode) if _builder is None else raw_vector_mode
        )
        self._builder = _builder
        self._semantic = _semantic

        # Validate core_mode
        if self.core_mode is not None and self.core_mode not in _VALID_CORE_MODES:
            raise ValueError(f"core_mode must be one of {_VALID_CORE_MODES}, got {self.core_mode}")

        if self.vector_mode is not None and self.vector_mode not in _VALID_VECTOR_MODES:
            raise ValueError(f"vector_mode must be one of {_VALID_VECTOR_MODES}, got {self.vector_mode}")

        if self.core_mode == "cube" and self.vector_mode is not None:
            raise ValueError('vector_mode cannot be set when core_mode="cube"')

        if self.core_mode is None and self.vector_mode is None and not kwargs:
            raise ValueError("scope requires at least one argument")

    def __enter__(self):
        if self._builder is None:
            raise RuntimeError("scope can only be used inside a Triton kernel")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
