import numpy as np
import torch
from vllm.distributed import get_dcp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpecKind,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend import envs
from vllm_ascend.core.private_circle_pool import is_private_circle_kv_cache_spec
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_world_size
from vllm_ascend.ops.triton.compute_slot_mapping import (
    _compute_slot_mapping_kernel,
    _next_power_of_2,
)


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_sizes: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        num_speculative_tokens: int = 0,
        kv_cache_group: KVCacheGroupSpec = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.dcp_world_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group
        is_mamba_group = (
            kv_cache_group is not None
            and hasattr(kv_cache_group, "kv_cache_spec")
            and get_kv_cache_spec_kind(kv_cache_group.kv_cache_spec) == KVCacheSpecKind.MAMBA
        )
        # The KV cache spec already provides the per-rank table capacity.
        # Mamba state is replicated across DCP ranks, not sharded then expanded.
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        self.physical_block_size = block_size
        self.is_mamba_group = is_mamba_group
        self.is_private_circle_group = False
        if kv_cache_group is not None:
            group_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                specs = group_spec.kv_cache_specs.values()
            else:
                specs = (group_spec,)
            self.is_private_circle_group = envs.VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL and any(
                is_private_circle_kv_cache_spec(spec) for spec in specs
            )

        # If kernel_sizes is None or [0], use physical block size (no splitting)
        if kernel_sizes is None or kernel_sizes == [0]:
            self.block_size = block_size
            self.logical_block_size = block_size
            self.blocks_per_phys_block = 1
            self.use_hybrid_blocks = False
        else:
            # Find the first kernel size that divides physical_block_size evenly
            selected_kernel_size = None
            for kernel_size in kernel_sizes:
                if kernel_size > 0 and self.physical_block_size % kernel_size == 0:
                    selected_kernel_size = kernel_size
                    break

            if selected_kernel_size is None:
                raise ValueError(
                    f"None of the kernel sizes {kernel_sizes} can divide "
                    f"physical block size {self.physical_block_size} evenly"
                )

            self.block_size = selected_kernel_size
            self.logical_block_size = selected_kernel_size
            self.blocks_per_phys_block = self.physical_block_size // self.logical_block_size
            if self.blocks_per_phys_block > 1:
                self.use_hybrid_blocks = True
            else:
                self.use_hybrid_blocks = False

        if self.use_hybrid_blocks:
            logical_table_size = max_num_blocks_per_req * self.blocks_per_phys_block
        else:
            logical_table_size = max_num_blocks_per_req

        duplicate_size = 1
        if self.dcp_world_size > 1:
            duplicate_size += num_speculative_tokens
        self.block_table = self._make_buffer(max_num_reqs * duplicate_size, logical_table_size, dtype=torch.int32)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        # MTP slot preparation appends up to num_speculative_tokens - 1
        # draft positions for every request beyond the scheduler token limit.
        num_mtp_draft_slots = max(num_speculative_tokens - 1, 0) * self.max_num_reqs
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens + num_mtp_draft_slots,
            dtype=torch.int32,
        )

        self.kernel_sizes = kernel_sizes
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids,
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        block_ids = np.array(block_ids)
        if self.use_hybrid_blocks:
            block_ids = self._convert_physical_to_logical_blocks(block_ids)

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]

        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
        self.num_blocks_per_row[row_idx] += num_blocks

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def clear_row(self, row_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table.np[tgt, :num_blocks] = self.block_table.np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src

        self.block_table.np[[src, tgt]] = self.block_table.np[[tgt, src]]

    def compute_private_circle_slot_mapping(
        self,
        positions: torch.Tensor,
        request_indices: torch.Tensor,
        allocation_ids: torch.Tensor,
        blocks_per_allocation: int,
    ) -> None:
        """Build absolute-position ring slots for request-private circle.

        ``allocation_ids`` maps each request row to the private allocation
        assigned by the coordinator.  This path intentionally does not use
        the shared block table or prefix-cache block hashes.
        """
        if blocks_per_allocation <= 0:
            raise ValueError("blocks_per_allocation must be positive")
        if positions.numel() != request_indices.numel():
            raise ValueError("positions and request_indices must have equal length")
        if allocation_ids.numel() == 0 and positions.numel() != 0:
            raise ValueError("allocation_ids cannot be empty for non-empty positions")
        block_size = self.physical_block_size
        req_allocations = allocation_ids.to(device=positions.device, dtype=torch.long)
        req = request_indices.to(device=positions.device, dtype=torch.long)
        pos = positions.to(dtype=torch.long)
        alloc = req_allocations.index_select(0, req)
        ring_block = torch.remainder(torch.div(pos, block_size, rounding_mode="floor"), blocks_per_allocation)
        slots = (1 + alloc * blocks_per_allocation + ring_block) * block_size + torch.remainder(pos, block_size)
        self.slot_mapping.gpu[: slots.numel()].copy_(slots.to(dtype=torch.int32))
        # Mirror the triton kernel's tail pad so padded (graph-replay) tokens
        # never write through stale ring slots.
        if slots.numel() < self.slot_mapping.gpu.numel():
            self.slot_mapping.gpu[slots.numel():].fill_(PAD_SLOT_ID)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_tokens = positions.shape[0]
        total_cp_world_size = self.dcp_world_size
        total_cp_rank = self.dcp_rank
        if self.dcp_world_size > 1:
            req_indices = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int32, device=query_start_loc.device),
                query_start_loc[1:] - query_start_loc[:-1],
                output_size=num_tokens,
            )
            self._compute_dcp_slot_mapping(req_indices, positions)
        else:
            TILE_BLOCK_SIZE = 1024
            kernel_kwargs = {
                "KV_CACHE_BLOCK_SIZE": self.physical_block_size,
                "BLOCKS_PER_KV_BLOCK": self.blocks_per_phys_block,
                "TOTAL_CP_WORLD_SIZE": total_cp_world_size,
                "TOTAL_CP_RANK": total_cp_rank,
                "CP_KV_CACHE_INTERLEAVE_SIZE": self.cp_kv_cache_interleave_size,
                "PAD_ID": PAD_SLOT_ID,
                "TILE_BLOCK_SIZE": TILE_BLOCK_SIZE,
                "BLOCK_TABLE_WINDOW_SIZE": _next_power_of_2(cdiv(TILE_BLOCK_SIZE, self.block_size) + 1),
            }

            _compute_slot_mapping_kernel[(num_reqs + 1,)](
                num_tokens,
                self.max_num_batched_tokens,
                query_start_loc,
                positions,
                self.block_table.gpu,
                self.block_table.gpu.stride(0),
                self.block_size,
                self.slot_mapping.gpu,
                **kernel_kwargs,
            )

    def compute_slot_mapping_draft(
        self,
        req_indices: np.ndarray | torch.Tensor,
        positions: np.ndarray | torch.Tensor,
    ) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.

        if self.dcp_world_size > 1:
            if not isinstance(req_indices, torch.Tensor):
                req_indices = torch.from_numpy(req_indices)
            if not isinstance(positions, torch.Tensor):
                positions = torch.from_numpy(positions)
            self._compute_dcp_slot_mapping(req_indices, positions)
        else:
            if isinstance(req_indices, torch.Tensor):
                if req_indices.device.type != "cpu":
                    raise ValueError("Device tensor inputs are only supported for CP draft slot mapping.")
                req_indices = req_indices.numpy()
            if isinstance(positions, torch.Tensor):
                if positions.device.type != "cpu":
                    raise ValueError("Device tensor inputs are only supported for CP draft slot mapping.")
                positions = positions.numpy()
            assert self.kernel_sizes is not None
            assert self.block_size == self.kernel_sizes[0]
            # IMPORTANT: In hybrid mode, positions are in logical block space,
            # but we need to map them to the correct logical block table indices
            logical_block_idx = positions // self.block_size

            # Account for the expanded logical table
            # (always needed with unified tensor)
            # Each physical block is split into multiple logical blocks
            # The logical table has been expanded to accommodate this
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx
            )

            block_offsets = positions % self.block_size
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping.np[: req_indices.shape[0]],
            )
            self.slot_mapping.copy_to_gpu(req_indices.shape[0])

    def _compute_dcp_slot_mapping(
        self,
        req_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # Note(hc): The DCP implement store kvcache with an interleave
        # style, the kvcache for the token whose token_idx is i is
        # always stored on the GPU whose dcp_rank equals the interleaved shard:

        # Use a "virtual block" which equals to world_size * block_size
        # for block_table_indices calculation.
        # virtual_block_size = self.block_size * self.dcp_world_size

        # IMPORTANT: In hybrid mode, positions are in logical block space,
        # but we need to map them to the correct logical block table indices
        # logical_block_idx = positions // virtual_block_size

        total_cp_world_size = self.dcp_world_size
        virtual_physical_block_size = self.physical_block_size * total_cp_world_size
        physical_block_idx = positions // virtual_physical_block_size
        virtual_block_offsets = positions % virtual_physical_block_size

        self.current_rank = self.dcp_rank
        mask = virtual_block_offsets // self.cp_kv_cache_interleave_size % total_cp_world_size == self.current_rank
        local_physical_offsets = (
            virtual_block_offsets
            // (total_cp_world_size * self.cp_kv_cache_interleave_size)
            * self.cp_kv_cache_interleave_size
            + virtual_block_offsets % self.cp_kv_cache_interleave_size
        )
        logical_block_idx = physical_block_idx * self.blocks_per_phys_block + (
            local_physical_offsets // self.block_size
        )

        block_table_indices = req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx

        block_offsets = local_physical_offsets % self.block_size

        if block_table_indices.device.type != "cpu":
            block_numbers = self.block_table.gpu.flatten()[block_table_indices]
            slot_mapping = block_numbers * self.block_size + block_offsets
            self.slot_mapping.gpu[: req_indices.shape[0]] = torch.where(mask, slot_mapping, -1)
        else:
            block_numbers = self.block_table.cpu.flatten()[block_table_indices]
            slot_mapping = block_numbers * self.block_size + block_offsets
            self.slot_mapping.cpu[: req_indices.shape[0]] = torch.where(mask, slot_mapping, -1)

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table.cpu.fill_(0)

    def _convert_physical_to_logical_blocks(self, physical_blocks: np.ndarray) -> np.ndarray:
        """Convert physical block IDs to logical block IDs."""
        if not self.use_hybrid_blocks:
            return physical_blocks

        # Create logical block IDs by splitting each physical block
        logical_blocks: list[int] = []
        for phys_block in physical_blocks:
            # Convert physical block to multiple logical blocks
            # Physical block 1 becomes logical blocks
            # [1*split_ratio, 1*split_ratio+1, ...]
            # But we need to account for the fact that block 0 is special
            base_logical = phys_block * self.blocks_per_phys_block
            logical_blocks.extend(range(base_logical, base_logical + self.blocks_per_phys_block))

        return np.array(logical_blocks, dtype=np.int32)

    def get_device_tensor(self, num_reqs: int | None = None) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        if num_reqs is not None:
            return self.block_table.gpu[:num_reqs]
        return self.block_table.gpu

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(self, *size: int | torch.SymInt, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size, dtype=dtype, device=self.device, pin_memory=self.pin_memory)


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        num_speculative_tokens: int = 0,
        max_num_blocks: list[int] | None = None,
        kernel_sizes: list[list[int]] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        kv_cache_groups: KVCacheGroupSpec = None,
    ) -> None:
        if kernel_sizes is None:
            kernel_sizes = [[0]] * len(block_sizes)
        # Ensure kernel_sizes matches block_sizes length
        elif len(kernel_sizes) == 1 and len(block_sizes) > 1:
            kernel_sizes = kernel_sizes * len(block_sizes)
        elif len(kernel_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match block_sizes length ({len(block_sizes)})"
            )

        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            dcp_world_size = get_decode_context_model_parallel_world_size()
            max_num_blocks = [cdiv(max_model_len, block_size * dcp_world_size) for block_size in block_sizes]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) must match block_sizes length ({len(block_sizes)})"
            )

        # Use zip to pair block_sizes with kernel_sizes one-to-one
        if kv_cache_groups is not None:
            self.block_tables = [
                BlockTable(
                    block_size,
                    max_num_reqs,
                    max_num_blocks_per_req,
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_size_list,
                    cp_kv_cache_interleave_size,
                    num_speculative_tokens,
                    kv_cache_group,
                )
                for block_size, kernel_size_list, max_num_blocks_per_req, kv_cache_group in zip(
                    block_sizes, kernel_sizes, max_num_blocks, kv_cache_groups
                )
            ]
        else:
            self.block_tables = [
                BlockTable(
                    block_size,
                    max_num_reqs,
                    max_num_blocks_per_req,
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_size_list,
                    cp_kv_cache_interleave_size,
                    num_speculative_tokens,
                )
                for block_size, kernel_size_list, max_num_blocks_per_req in zip(
                    block_sizes, kernel_sizes, max_num_blocks
                )
            ]

        self.private_circle_allocation_ids = torch.full(
            (max_num_reqs,), -1, dtype=torch.int64, device="cpu",
            pin_memory=pin_memory)
        self.private_circle_allocation_ids_gpu = None  # lazily created device mirror
        self.private_circle_blocks_per_allocation = 0
        self.private_circle_window_size = 0
        # Visible token span the ring table must mark valid on the left. This
        # is the ring capacity (``window_size - 1 + in_flight_tokens``): the
        # oldest position any reader references is max(target causal window
        # front, DSpark proposal block front) = ``seq_len - window -
        # in_flight + 1`` (worst case: all drafts rejected), so older
        # positions must not fall outside ``first_page``.
        self.private_circle_ring_window_tokens = 0
        # Right-edge extension for the ring table's logical_count: the DSpark
        # drafter resolves its proposal query block (positions up to
        # ``seq_len + in_flight - 1`` under full acceptance) through this
        # step's table before it is rebuilt, so those columns must already be
        # valid ring pages.
        self.private_circle_ring_in_flight_tokens = 1
        self._private_circle_installed_this_step = False

    def set_private_circle_allocations(
        self,
        allocation_ids: np.ndarray,
        blocks_per_allocation: int,
        num_allocations: int,
        private_num_blocks: int,
        window_size: int = 0,
        ring_window_tokens: int = 0,
        ring_in_flight_tokens: int = 1,
    ) -> None:
        """Install private-circle allocation metadata for device-side table build.

        Writes the allocation IDs to the pinned CPU buffer and issues a
        non-blocking H2D to the persistent GPU mirror. The ring block table
        itself is built on-device by ``build_private_circle_ring_table``
        (called from ``_prepare_inputs`` after seq_lens correction), using
        each group's own ``physical_block_size``.

        ``window_size`` is the target SWA window. ``ring_window_tokens`` is
        the ring's visible span (its capacity) and drives the table's left
        edge (``first_page``); it defaults to ``window_size`` for callers
        that only need the target window. ``ring_in_flight_tokens``
        (``1 + num_speculative_tokens``) extends the table's right edge
        (``logical_count``) past the last scheduled token so the drafter's
        proposal query block resolves to ring pages instead of the null
        block; ``1`` (no drafting) keeps the legacy right end.
        """
        if len(allocation_ids) > self.private_circle_allocation_ids.numel():
            raise ValueError("too many private circle allocations")
        if blocks_per_allocation <= 0 or num_allocations <= 0:
            raise ValueError("private circle allocation dimensions must be positive")
        expected_num_blocks = 1 + num_allocations * blocks_per_allocation
        if private_num_blocks != expected_num_blocks:
            raise ValueError(
                "private circle block count does not match allocation layout: "
                f"{private_num_blocks} != {expected_num_blocks}"
            )
        if np.any(allocation_ids >= num_allocations):
            raise IndexError("private circle allocation handle is out of range")

        self.private_circle_allocation_ids.fill_(-1)
        if len(allocation_ids):
            self.private_circle_allocation_ids[:len(allocation_ids)] = torch.as_tensor(
                allocation_ids, dtype=torch.int64)
        self.private_circle_blocks_per_allocation = blocks_per_allocation
        self.private_circle_window_size = window_size
        self.private_circle_ring_window_tokens = ring_window_tokens or window_size
        self.private_circle_ring_in_flight_tokens = max(1, ring_in_flight_tokens)
        self._private_circle_installed_this_step = True

        # H2D to persistent GPU mirror (non_blocking from pinned).
        if self.private_circle_allocation_ids_gpu is None:
            # Derive device from the block table group's GPU buffer (first
            # PC group), falling back to the allocation buffer's device.
            # For NPU-only codebase this is always an NPU device.
            device = (
                self.block_tables[0].block_table.gpu.device
                if self.block_tables
                else self.private_circle_allocation_ids.device
            )
            self.private_circle_allocation_ids_gpu = torch.full(
                (self.private_circle_allocation_ids.numel(),),
                -1,
                dtype=torch.int64,
                device=device,
            )
        self.private_circle_allocation_ids_gpu.copy_(
            self.private_circle_allocation_ids, non_blocking=True
        )

    def build_private_circle_ring_table_device(
        self,
        seq_lens: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Build the ring block table on device for all PC groups."""
        from vllm_ascend.ops.triton.private_circle_ring_table import (
            build_private_circle_ring_table,
        )

        # Consume-on-read: the flag is set by set_private_circle_allocations
        # (in _update_states) and cleared here after the first build call.
        # This ensures each step's build uses only the allocation metadata
        # installed by this step's _update_states, never a previous step's.
        installed = self._private_circle_installed_this_step
        self._private_circle_installed_this_step = False
        if (
            self.private_circle_blocks_per_allocation <= 0
            or self.private_circle_allocation_ids_gpu is None
            or not self.private_circle_ring_window_tokens
            or not installed
        ):
            return
        for table in self.block_tables:
            if not table.is_private_circle_group:
                continue
            build_private_circle_ring_table(
                seq_lens,
                self.private_circle_allocation_ids_gpu,
                table.block_table.gpu,
                num_reqs,
                window_size=self.private_circle_ring_window_tokens,
                block_size=table.physical_block_size,
                blocks_per_allocation=self.private_circle_blocks_per_allocation,
                in_flight_tokens=self.private_circle_ring_in_flight_tokens,
            )

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def mask_shared_slots_for_private_circle_bounded_replay(
        self,
        positions: torch.Tensor,
        request_indices: torch.Tensor,
        persistent_starts: torch.Tensor,
    ) -> None:
        """Prevent [R, H) from updating any shared KV-cache group."""
        if positions.numel() != request_indices.numel():
            raise ValueError(
                "positions and request_indices must have equal length"
            )
        req = request_indices.to(device=positions.device, dtype=torch.long)
        starts = persistent_starts.to(
            device=positions.device, dtype=torch.long
        ).index_select(0, req)
        bounded_replay_mask = (starts >= 0) & (positions.to(torch.long) < starts)
        for block_table in self.block_tables:
            if block_table.is_private_circle_group or block_table.is_mamba_group:
                continue
            block_table.slot_mapping.gpu[: positions.numel()].masked_fill_(
                bounded_replay_mask, -1
            )

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        for i, block_table in enumerate(self.block_tables):
            if block_table.is_mamba_group:
                continue
            if block_table.is_private_circle_group:
                if self.private_circle_blocks_per_allocation <= 0:
                    raise RuntimeError("private circle allocation metadata is missing")
                req_indices = torch.repeat_interleave(
                    torch.arange(num_reqs, dtype=torch.int64,
                                 device=positions.device),
                    query_start_loc[1:] - query_start_loc[:-1],
                    output_size=positions.numel(),
                )
                block_table.compute_private_circle_slot_mapping(
                    positions,
                    req_indices,
                    (self.private_circle_allocation_ids_gpu[:num_reqs]
                     if self.private_circle_allocation_ids_gpu is not None
                     else self.private_circle_allocation_ids[:num_reqs]),
                    self.private_circle_blocks_per_allocation,
                )
                continue
            if positions_compressed_list and req_indices_compressed_list:
                block_table.compute_slot_mapping_draft(req_indices_compressed_list[i], positions_compressed_list[i])
            else:
                block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def compute_slot_mapping_draft(
        self,
        req_indices: np.ndarray | torch.Tensor,
        positions: np.ndarray | torch.Tensor,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        for i, block_table in enumerate(self.block_tables):
            if block_table.is_mamba_group:
                continue
            if block_table.is_private_circle_group:
                if self.private_circle_blocks_per_allocation <= 0:
                    raise RuntimeError("private circle allocation metadata is missing")
                req_tensor = (req_indices if isinstance(req_indices, torch.Tensor)
                              else torch.from_numpy(req_indices))
                pos_tensor = (positions if isinstance(positions, torch.Tensor)
                              else torch.from_numpy(positions))
                block_table.compute_private_circle_slot_mapping(
                    pos_tensor, req_tensor,
                    (self.private_circle_allocation_ids_gpu
                     if self.private_circle_allocation_ids_gpu is not None
                     else self.private_circle_allocation_ids),
                    self.private_circle_blocks_per_allocation)
                continue
            if positions_compressed_list and req_indices_compressed_list:
                block_table.compute_slot_mapping_draft(req_indices_compressed_list[i], positions_compressed_list[i])
            else:
                block_table.compute_slot_mapping_draft(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            if block_table.is_private_circle_group:
                # PC group's table is built on-device by the ring table
                # kernel; no host-to-device commit needed.
                continue
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
