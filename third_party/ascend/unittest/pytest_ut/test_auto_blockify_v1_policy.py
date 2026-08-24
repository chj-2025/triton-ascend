# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from triton.backends.ascend.compiler import _resolve_auto_blockify_v1_policy

SAFE_TTIR = "module { tt.func public @safe() { tt.return } }"
ATOMIC_TTIR = "module { tt.func public @atomic() { %0 = tt.atomic_rmw add } }"


@pytest.mark.parametrize(
    "env_enabled,option,expected,source",
    [
        (True, None, True, "TRITON_ALL_BLOCKS_PARALLEL"),
        (False, None, False, "TRITON_ALL_BLOCKS_PARALLEL"),
        (True, False, False, "option"),
        (False, True, True, "option"),
    ],
)
def test_auto_blockify_v1_enable_policy(env_enabled, option, expected, source):
    metadata = {}
    opt = SimpleNamespace(enable_auto_blockify=option)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=env_enabled):
        assert _resolve_auto_blockify_v1_policy(SAFE_TTIR, metadata, opt) is expected
    assert metadata["auto_blockify_v1_enabled"] is expected
    assert metadata["auto_blockify_v1_selection_source"] == source


def test_auto_blockify_v1_blacklist_disables_both_implementations():
    metadata = {}
    opt = SimpleNamespace(enable_auto_blockify=None)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=True), \
         patch("triton.backends.ascend.compiler._warn_auto_blockify_disabled") as warn:
        assert not _resolve_auto_blockify_v1_policy(ATOMIC_TTIR, metadata, opt)
    assert metadata["has_auto_blockify_blacklist_op"]
    warn.assert_called_once()


def test_auto_blockify_v1_explicit_blacklist_override_is_preserved():
    metadata = {"has_auto_blockify_blacklist_op": False}
    opt = SimpleNamespace(enable_auto_blockify=True)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=False):
        assert _resolve_auto_blockify_v1_policy(ATOMIC_TTIR, metadata, opt)
