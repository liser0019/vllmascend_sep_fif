from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _a5_operator_block() -> str:
    build_script = (_REPO_ROOT / "csrc/build_aclnn.sh").read_text(encoding="utf-8")
    return build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend950 ]]', maxsplit=1)[1].split("else", maxsplit=1)[0]


def test_a5_excludes_chunk_kernels_without_objects():
    a5_ops = _a5_operator_block()

    assert '"chunk_fwd_o"' not in a5_ops
    assert '"chunk_gated_delta_rule_fwd_h"' not in a5_ops
    assert '"chunk_kda_fwd"' not in a5_ops


def test_other_soc_operator_lists_keep_supported_chunk_kernels():
    build_script = (_REPO_ROOT / "csrc/build_aclnn.sh").read_text(encoding="utf-8")
    pre_a5_blocks = build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend950 ]]', maxsplit=1)[0]

    assert '"chunk_fwd_o"' in pre_a5_blocks
    assert '"chunk_gated_delta_rule_fwd_h"' in pre_a5_blocks
    assert '"chunk_kda_fwd"' in pre_a5_blocks


def test_quant_indexer_dependency_is_stable_and_opsbase_is_linked():
    quant_cmake = (_REPO_ROOT / "csrc/attention/quant_lightning_indexer_v2/op_host/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    custom_build = (_REPO_ROOT / "csrc/cmake/custom_build.cmake").read_text(encoding="utf-8")

    dependency = 'quant_lightning_indexer_v2_depends\n    "attention/lightning_indexer_v2"'
    assert dependency in quant_cmake
    assert "CACHE STRING" in quant_cmake
    assert "FORCE" in quant_cmake

    package_branch = custom_build.split(
        "if(BUILD_WITH_3_8_PACKAGE)\ntarget_link_libraries(\n    cust_opmaster", maxsplit=1
    )[1].split("else ()", maxsplit=1)[0]
    assert "$<$<TARGET_EXISTS:opsbase>:opsbase>" in package_branch
