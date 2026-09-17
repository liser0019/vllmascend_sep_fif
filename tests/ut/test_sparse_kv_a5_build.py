# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ascend950_ops() -> set[str]:
    build_script = (_REPO_ROOT / "csrc/build_aclnn.sh").read_text(encoding="utf-8")
    match = re.search(
        r'elif \[\[ "\$SOC_VERSION" =~ \^ascend950 \]\]; then(?P<body>.*?)\nelse\n',
        build_script,
        re.DOTALL,
    )
    assert match is not None
    return set(re.findall(r'^\s+"([a-z0-9_]+)"$', match.group("body"), re.MULTILINE))


def test_a5_build_selects_sparse_kv_operators_and_skips_unsupported_chunk_ops():
    selected_ops = _ascend950_ops()

    assert {"sparse_kv_plan", "sparse_kv_transfer"} <= selected_ops
    assert not {"chunk_fwd_o", "chunk_gated_delta_rule_fwd_h", "chunk_kda_fwd"} & selected_ops
    for op_name in ("sparse_kv_plan", "sparse_kv_transfer"):
        assert (_REPO_ROOT / "csrc/attention" / op_name / "op_kernel").is_dir()


def test_sparse_kv_a5_sources_use_cann_91_interfaces():
    plan_tiling = (_REPO_ROOT / "csrc/attention/sparse_kv_plan/op_host/sparse_kv_plan_tiling.cpp").read_text(
        encoding="utf-8"
    )
    transfer_tiling = (
        _REPO_ROOT / "csrc/attention/sparse_kv_transfer/op_host/sparse_kv_transfer_tiling.cpp"
    ).read_text(encoding="utf-8")
    plan_kernel = (_REPO_ROOT / "csrc/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_apt.cpp").read_text(
        encoding="utf-8"
    )

    assert plan_tiling.count("const gert::StorageShape*") == 3
    assert transfer_tiling.count("const gert::StorageShape*") == 2
    assert 'extern "C" __global__ __aicore__ void sparse_kv_plan' in plan_kernel
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);" in plan_kernel
