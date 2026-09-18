import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.utils import AscendDeviceType


class TestSparseKVCustomOpLoading(unittest.TestCase):
    def tearDown(self):
        manager_module._SPARSE_KV_OFFLOAD_OPS = None

    @staticmethod
    def _all_sparse_ops():
        return SimpleNamespace(**{name: object() for name in manager_module._SPARSE_KV_OFFLOAD_OP_NAMES})

    def test_a5_loads_native_extension_once_without_enabling_all_custom_ops(self):
        sparse_ops = self._all_sparse_ops()
        with (
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(manager_module, "bootstrap_custom_op_env") as bootstrap,
            patch.object(manager_module, "enable_custom_op") as enable_custom_op,
            patch.object(manager_module.importlib, "import_module") as import_module,
            patch.object(manager_module.torch.ops, "_C_ascend", sparse_ops),
        ):
            self.assertIs(manager_module._sparse_kv_ops(), sparse_ops)
            self.assertIs(manager_module._sparse_kv_ops(), sparse_ops)

        bootstrap.assert_called_once_with()
        import_module.assert_called_once_with("vllm_ascend.vllm_ascend_C")
        enable_custom_op.assert_not_called()

    def test_a5_preserves_native_extension_import_error(self):
        import_error = ImportError("extension is unavailable")
        with (
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(manager_module, "bootstrap_custom_op_env"),
            patch.object(manager_module.importlib, "import_module", side_effect=import_error),
            self.assertRaisesRegex(RuntimeError, "Failed to load A5 Sparse KV custom ops") as raised,
        ):
            manager_module._sparse_kv_ops()

        self.assertIs(raised.exception.__cause__, import_error)

    def test_a5_reports_missing_sparse_ops(self):
        sparse_ops = SimpleNamespace(sparse_kv_restore_bfloat16_tensor=object())
        with (
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(manager_module, "bootstrap_custom_op_env"),
            patch.object(manager_module.importlib, "import_module"),
            patch.object(manager_module.torch.ops, "_C_ascend", sparse_ops),
            self.assertRaisesRegex(RuntimeError, "sparse_kv_warmup_lru_resident_threads"),
        ):
            manager_module._sparse_kv_ops()

    def test_non_a5_uses_general_custom_op_loader(self):
        sparse_ops = self._all_sparse_ops()
        with (
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A3),
            patch.object(manager_module, "bootstrap_custom_op_env") as bootstrap,
            patch.object(manager_module, "enable_custom_op", return_value=True) as enable_custom_op,
            patch.object(manager_module.importlib, "import_module") as import_module,
            patch.object(manager_module.torch.ops, "_C_ascend", sparse_ops),
        ):
            self.assertIs(manager_module._sparse_kv_ops(), sparse_ops)

        enable_custom_op.assert_called_once_with()
        bootstrap.assert_not_called()
        import_module.assert_not_called()


class _FakeKVCacheSpec:
    def __init__(
        self,
        *,
        page_size_bytes,
        max_blocks_per_request,
        store_on_host,
        block_size=128,
    ):
        self.page_size_bytes = page_size_bytes
        self.max_blocks_per_request = max_blocks_per_request
        self.store_on_host = store_on_host
        self.block_size = block_size

    def max_memory_usage_bytes(self, _vllm_config):
        return self.max_blocks_per_request * self.page_size_bytes


def _make_memory_plan_inputs(max_num_seqs=2):
    specs = {
        "host.0": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "host.1": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "device.0": _FakeKVCacheSpec(
            page_size_bytes=512,
            max_blocks_per_request=100,
            store_on_host=False,
        ),
    }
    vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
    alignment_reserve = 2 * manager_module._CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    return specs, vllm_config, alignment_reserve


class TestSparseKVOffloadMemoryPlanning(unittest.TestCase):
    def test_non_a3_is_rejected(self):
        with (
            patch.object(manager_module, "_SPARSE_KV_OFFLOAD_MANAGER", None),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A2),
            self.assertRaisesRegex(RuntimeError, "supports Atlas A3 and A5"),
        ):
            manager_module.init_sparse_kv_offload_manager(None, None, None)

    def test_memory_plan_is_limited_by_active_workload(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.npu_limit_blocks, 1000)
        self.assertEqual(budget.dram_limit_blocks, 1000)
        self.assertEqual(budget.workload_limit_blocks, 201)
        self.assertEqual(budget.final_num_blocks, 201)
        self.assertEqual(budget.final_planner_bytes, 201 * (2048 + 512))
        self.assertEqual(budget.planned_host_bytes, 201 * 2048)
        self.assertEqual(budget.planned_device_bytes, 201 * 512)
        self.assertEqual(budget.limiting_factor, "workload")

    def test_memory_plan_is_limited_by_dram_capacity(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 150 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.dram_limit_blocks, 150)
        self.assertEqual(budget.final_num_blocks, 150)
        self.assertEqual(budget.limiting_factor, "dram")

    def test_warning_when_dram_budget_caps_npu_utilization(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with self.assertLogs(manager_module.logger, level="WARNING") as logs:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 500 * 2048,
                keep_device_kv_cache=False,
            )
        self.assertTrue(any("dram_size_per_dp_GB" in line for line in logs.output))

    def test_no_warning_when_dram_budget_not_below_npu(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with patch.object(manager_module.logger, "warning_once") as mock_warn:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 1000 * 2048,
                keep_device_kv_cache=False,
            )
        mock_warn.assert_not_called()

    def test_keep_device_cache_counts_full_npu_page(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        total_page_size = 2048 + 512

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=125 * total_page_size,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=True,
        )

        self.assertEqual(budget.npu_limit_blocks, 125)
        self.assertEqual(budget.final_num_blocks, 125)
        self.assertEqual(budget.planned_device_bytes, 125 * total_page_size)
        self.assertEqual(budget.limiting_factor, "npu")

    def test_non_positive_capacity_produces_zero_blocks(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        for available_device_memory, dram_limit, limiting_factor in (
            (-1, alignment_reserve + 1000 * 2048, "npu"),
            (1000 * 512, alignment_reserve - 1, "dram"),
        ):
            with self.subTest(limiting_factor=limiting_factor):
                budget = plan_sparse_kv_offload_memory(
                    kv_cache_spec=specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=available_device_memory,
                    dram_limit_bytes=dram_limit,
                    keep_device_kv_cache=False,
                )
                self.assertEqual(budget.final_num_blocks, 0)
                self.assertEqual(budget.final_planner_bytes, 0)
                self.assertEqual(budget.limiting_factor, limiting_factor)

    def test_memory_plan_rejects_invalid_spec_layouts(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        invalid_cases = (
            ({"device.0": specs["device.0"]}, "at least one host"),
            ({"host.0": specs["host.0"]}, "at least one device"),
            (
                {
                    **specs,
                    "device.1": _FakeKVCacheSpec(
                        page_size_bytes=512,
                        max_blocks_per_request=100,
                        store_on_host=False,
                        block_size=64,
                    ),
                },
                "one shared block size",
            ),
        )

        for invalid_specs, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                plan_sparse_kv_offload_memory(
                    kv_cache_spec=invalid_specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=1000 * 512,
                    dram_limit_bytes=alignment_reserve + 1000 * 2048,
                    keep_device_kv_cache=False,
                )

    def test_cpu_pool_size_includes_per_layer_alignment_reserve(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[name],
                    kv_cache_spec=spec,
                )
                for name, spec in specs.items()
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_supports_uniform_specs(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        uniform_specs = UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs=specs,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=list(specs),
                    kv_cache_spec=uniform_specs,
                )
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_rejects_missing_host_specs(self):
        specs, _, _ = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["device.0"],
                    kv_cache_spec=specs["device.0"],
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "host-resident"):
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config)

    def _make_manager_init_inputs(self, dram_size_per_dp_gb=1):
        spec = _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[SimpleNamespace(layer_names=["host.0"], kv_cache_spec=spec)],
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                get_num_layers=MagicMock(return_value=1),
                max_model_len=128,
            ),
            parallel_config=SimpleNamespace(),
            scheduler_config=SimpleNamespace(
                max_num_seqs=1,
                max_num_batched_tokens=1,
            ),
            speculative_config=None,
        )
        offload_config = SimpleNamespace(
            topk_buffer_size=1,
            topk=1,
            use_fused_overlap=False,
            dram_size_per_dp_GB=dram_size_per_dp_gb,
        )
        return vllm_config, kv_cache_config, offload_config

    def test_manager_initializes_offload_with_planned_pool_size(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        planned_pool_size = 4096
        for rank, expected_alloc_size in ((0, planned_pool_size), (1, 0)):
            with self.subTest(rank=rank):
                offload_backend = SimpleNamespace(
                    OffloadConfig=lambda: SimpleNamespace(),
                    Scene=SimpleNamespace(SHARED="shared"),
                    initialize=MagicMock(return_value=0),
                )
                tp_group = SimpleNamespace(barrier=MagicMock())

                with (
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_rank",
                        return_value=rank,
                    ),
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_world_size",
                        return_value=2,
                    ),
                    patch.object(
                        manager_module,
                        "get_tp_group",
                        return_value=tp_group,
                    ),
                    patch.object(
                        manager_module,
                        "get_sparse_kv_offload_cpu_pool_size_bytes",
                        return_value=planned_pool_size,
                    ),
                    patch.object(
                        manager_module,
                        "offload",
                        offload_backend,
                        create=True,
                    ),
                    patch.object(
                        manager_module.torch,
                        "zeros",
                        return_value=MagicMock(),
                    ),
                    patch.object(
                        manager_module.torch,
                        "empty",
                        return_value=MagicMock(),
                    ),
                    patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
                    patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A3),
                ):
                    SparseKVOffloadManager(
                        vllm_config,
                        kv_cache_config,
                        offload_config,
                    )

                initialized_config = offload_backend.initialize.call_args.args[0]
                self.assertEqual(
                    initialized_config.reserve_size,
                    planned_pool_size,
                )
                self.assertEqual(
                    initialized_config.alloc_size,
                    expected_alloc_size,
                )
                self.assertEqual(initialized_config.world_size, 2)
                self.assertEqual(initialized_config.rank_id, rank)
                tp_group.barrier.assert_called_once_with()

    def test_a5_manager_uses_rank_local_urma_pool(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        vllm_config.kv_transfer_config = SimpleNamespace(
            kv_connector_extra_config={"memfabric_transfer_protocol": "device_urma"}
        )
        planned_pool_size = 4096
        offload_backend = SimpleNamespace(
            OffloadConfig=lambda: SimpleNamespace(),
            OFFLOAD_FLAG_GIANT_PAGE=1,
            Scene=SimpleNamespace(LOCAL="local", SHARED="shared"),
            get_dva=MagicMock(return_value=0x2000),
            initialize=MagicMock(return_value=0),
        )
        tp_group = SimpleNamespace(barrier=MagicMock())

        with (
            patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=1),
            patch.object(manager_module, "get_tensor_model_parallel_world_size", return_value=2),
            patch.object(manager_module, "get_tp_group", return_value=tp_group),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=planned_pool_size,
            ),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        ):
            manager = SparseKVOffloadManager(vllm_config, kv_cache_config, offload_config)

        initialized_config = offload_backend.initialize.call_args.args[0]
        self.assertTrue(manager.rank_local_host_pool)
        self.assertEqual(initialized_config.reserve_size, planned_pool_size)
        self.assertEqual(initialized_config.alloc_size, planned_pool_size)
        self.assertEqual(initialized_config.world_size, 1)
        self.assertEqual(initialized_config.rank_id, 0)
        self.assertEqual(initialized_config.scene, "local")
        self.assertEqual(initialized_config.flags, 1)
        tp_group.barrier.assert_called_once_with()

    def test_a5_manager_requires_device_urma_for_pd(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        vllm_config.kv_transfer_config = SimpleNamespace(
            kv_connector_extra_config={"memfabric_transfer_protocol": "sdma"}
        )
        offload_backend = SimpleNamespace(
            OffloadConfig=lambda: SimpleNamespace(),
            OFFLOAD_FLAG_GIANT_PAGE=1,
            Scene=SimpleNamespace(LOCAL="local", SHARED="shared"),
            get_dva=MagicMock(return_value=0x2000),
            initialize=MagicMock(return_value=0),
        )

        with (
            patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=0),
            patch.object(manager_module, "get_tensor_model_parallel_world_size", return_value=1),
            patch.object(manager_module, "get_tp_group", return_value=SimpleNamespace()),
            patch.object(manager_module, "get_sparse_kv_offload_cpu_pool_size_bytes", return_value=4096),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            self.assertRaisesRegex(ValueError, "device_urma"),
        ):
            SparseKVOffloadManager(vllm_config, kv_cache_config, offload_config)

        offload_backend.initialize.assert_not_called()

    def test_rank_local_pool_keeps_host_and_device_addresses_separate(self):
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.rank_local_host_pool = True
        manager.num_layers = 1
        manager.kv_cache_config = SimpleNamespace(num_blocks=4)
        manager.k_caches_cpu = [SimpleNamespace(data_ptr=lambda: 0x1000, numel=lambda: 40, element_size=lambda: 2)]
        manager.v_caches_cpu = [SimpleNamespace(data_ptr=lambda: 0x2000, numel=lambda: 20, element_size=lambda: 2)]
        offload_backend = SimpleNamespace(get_dva=MagicMock(side_effect=[0xA000, 0xB000]))

        with patch.object(manager_module, "offload", offload_backend, create=True):
            manager._initialize_pool_address_tables()

        self.assertEqual(manager.hvas_k_bases, [0x1000])
        self.assertEqual(manager.hvas_v_bases, [0x2000])
        self.assertEqual(manager.dvas_k_bases, [0xA000])
        self.assertEqual(manager.dvas_v_bases, [0xB000])
        self.assertEqual(manager.cpu_block_lens, [(20, 10)])

    def test_manager_rejects_pool_larger_than_dram_limit(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        offload_backend = SimpleNamespace(
            OffloadConfig=MagicMock(),
            Scene=SimpleNamespace(SHARED="shared"),
            initialize=MagicMock(return_value=0),
        )

        with (
            patch.object(
                manager_module,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch.object(
                manager_module,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch.object(
                manager_module,
                "get_tp_group",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=(1 << 30) + 1,
            ),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A3),
            self.assertRaisesRegex(ValueError, "exceeds DRAM limit"),
        ):
            SparseKVOffloadManager(
                vllm_config,
                kv_cache_config,
                offload_config,
            )

        offload_backend.OffloadConfig.assert_not_called()
        offload_backend.initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
