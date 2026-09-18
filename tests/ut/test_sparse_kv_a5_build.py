# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ops_config_module():
    util_dir = _REPO_ROOT / "csrc/cmake/scripts/util"
    module_path = util_dir / "ascendc_ops_config.py"
    spec = importlib.util.spec_from_file_location("test_ascendc_ops_config", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(util_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(util_dir))
    return module


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
    adapter = (_REPO_ROOT / "csrc/attention/sparse_kv_plan/sparse_kv_lru_torch_adpt.h").read_text(encoding="utf-8")

    assert plan_tiling.count("const gert::StorageShape*") == 3
    assert transfer_tiling.count("const gert::StorageShape*") == 2
    assert 'extern "C" __global__ __aicore__ void sparse_kv_plan' in plan_kernel
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);" in plan_kernel
    assert '#include "../../aclnn_torch_adapter/op_api_common.h"' in adapter


def test_cann_build_links_opsbase_and_stages_quant_indexer_dependency():
    custom_build = (_REPO_ROOT / "csrc/cmake/custom_build.cmake").read_text(encoding="utf-8")
    quant_cmake = (_REPO_ROOT / "csrc/attention/quant_lightning_indexer_v2/op_host/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    package_branch = re.search(
        r"if\(BUILD_WITH_3_8_PACKAGE\)\s+target_link_libraries\(\s+cust_opmaster(?P<body>.*?)\n\)",
        custom_build,
        re.DOTALL,
    )

    assert package_branch is not None
    assert "$<$<TARGET_EXISTS:opsbase>:opsbase>" in package_branch.group("body")
    assert '"attention/lightning_indexer_v2"' in quant_cmake
    assert "CACHE STRING" in quant_cmake
    assert "FORCE" in quant_cmake


def test_incremental_staging_tracks_sources_and_passes_selected_ops():
    cmake = (_REPO_ROOT / "csrc/cmake/func.cmake").read_text(encoding="utf-8")

    assert "GLOB_RECURSE SRC_COPY_DEP_FILES" in cmake
    assert "CONFIGURE_DEPENDS" in cmake
    assert "DEPENDS ${SRC_COPY_DEP_FILES}" in cmake
    assert "--selected-ops ${_selected_op_dirs}" in cmake


def test_ops_config_ignores_stale_unselected_directories(tmp_path):
    ops_config = _load_ops_config_module()
    selected_dir = tmp_path / "selected_op"
    selected_dir.mkdir()
    selected_json = selected_dir / "kernel.json"
    selected_json.write_text("{}", encoding="utf-8")
    stale_dir = tmp_path / "stale_op"
    stale_dir.mkdir()
    (stale_dir / "stale.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stale_empty_op").mkdir()

    ops_config.check_single_op_is_void(str(tmp_path), ["selected_op"])
    selected_files = ops_config.get_selected_suffix_files(str(tmp_path), ".json", ["selected_op"])

    assert selected_files == [str(selected_json)]


def test_ops_config_rejects_selected_operator_without_output(tmp_path):
    ops_config = _load_ops_config_module()
    (tmp_path / "selected_op").mkdir()

    with pytest.raises(SystemExit):
        ops_config.check_single_op_is_void(str(tmp_path), ["selected_op"])


def test_ops_config_accepts_absolute_selected_path_and_preserves_full_check(tmp_path):
    ops_config = _load_ops_config_module()
    selected_dir = tmp_path / "selected_op"
    selected_dir.mkdir()
    selected_json = selected_dir / "kernel.json"
    selected_json.write_text("{}", encoding="utf-8")
    (tmp_path / "stale_empty_op").mkdir()

    selected_files = ops_config.get_selected_suffix_files(str(tmp_path), ".json", [str(selected_dir)])
    assert selected_files == [str(selected_json)]
    ops_config.check_single_op_is_void(str(tmp_path), [str(selected_dir)])

    with pytest.raises(SystemExit):
        ops_config.check_single_op_is_void(str(tmp_path))
