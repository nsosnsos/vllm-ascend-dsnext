# SPDX-License-Identifier: MIT
# Adapted from the DeepSeek V4.1 reference inference/engram.py.
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sympy import isprime
from torch import nn
from vllm.logger import logger


def valid_engram_token_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Exclude the complete V4.1 image region from n-gram history."""
    return (input_ids != image_token_id) & (input_ids != image_pad_token_id)


def build_lookback_token_ids(
    histories: Sequence[Sequence[int]],
    chunk_starts: np.ndarray,
    depth: int,
    image_token_id: int,
    image_pad_token_id: int,
    image_spans: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build newest-first lookback from authoritative request histories."""
    chunk_starts = np.asarray(chunk_starts)
    if chunk_starts.shape != (len(histories),):
        raise ValueError("Engram token history and chunk starts have incompatible shapes")
    if depth < 1:
        raise ValueError("Engram lookback depth must be positive")
    if image_spans is not None and len(image_spans) != len(histories):
        raise ValueError("Engram image spans and token histories have incompatible shapes")

    lookback = np.full((len(histories), depth), -1, dtype=np.int64)
    dead = np.ones((len(histories), depth), dtype=np.bool_)
    for req_index, (history, chunk_start) in enumerate(zip(histories, chunk_starts, strict=True)):
        chunk_start = int(chunk_start)
        if chunk_start < 0 or chunk_start > len(history):
            logger.warning(
                "FOR-ENGRAM input history missing: request_index=%d history_length=%d chunk_start=%d",
                req_index,
                len(history),
                chunk_start,
            )
            raise ValueError(f"Engram request {req_index} is missing authoritative text token history")
        spans = () if image_spans is None else image_spans[req_index]
        for offset in range(depth):
            position = chunk_start - offset - 1
            if position < 0:
                continue
            token_id = int(history[position])
            lookback[req_index, offset] = token_id
            in_image_span = any(start <= position < start + length for start, length in spans)
            is_image_token = token_id in (image_token_id, image_pad_token_id)
            dead[req_index, offset] = token_id < 0 or is_image_token or in_image_span
    return lookback, dead


class EngramTokenSequence(Sequence[int]):
    """Zero-copy prompt plus accepted-output view used for Engram history."""

    def __init__(self, prompt: Sequence[int], output: Sequence[int]):
        self.prompt = prompt
        self.output = output

    def __len__(self) -> int:
        return len(self.prompt) + len(self.output)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.prompt):
            return self.prompt[index]
        return self.output[index - len(self.prompt)]


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike
    collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash
    the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters
    beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead
    # of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no
    # clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw
            # form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """Derive one multiplier per (layer, lookback) from a per-layer RNG.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position uses `max_ngram_size - 1` n-grams, each split over `n_heads`.
    Each (n-gram size, head) pair owns its own prime-sized bucket range in the
    layer's table; the primes are drawn in order and never reused, which keeps the
    ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        if max_ngram_size < 2 or n_heads < 1:
            raise ValueError("Engram requires max_ngram_size >= 2 and n_heads >= 1")
        num_embeddings = tuple(args.engram_num_embeddings)
        if len(num_embeddings) != len(layer_ids):
            raise ValueError("Engram num_embeddings must have one table size per layer")
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        expected_rows = tuple(sum(size for ngram in layer for size in ngram) for layer in primes)
        if any(actual < expected for actual, expected in zip(num_embeddings, expected_rows)):
            raise ValueError(
                "Engram table is smaller than its configured hash buckets: "
                f"checkpoint={num_embeddings}, minimum={expected_rows}"
            )
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class NgramHashState(nn.Module):
    """Stateless Engram hash over the current chunks and explicit history.

    ``lookback_token_ids`` is newest-first and contains the tokens immediately
    before each request's current chunk. This is the same contract for prefill,
    decode, prefix-cache hits, and P/D recovery, so no KV-page state is needed.
    """

    DEAD = -1

    def __init__(self, config, tokenizer):
        super().__init__()
        layout = EngramLayout.from_args(config)
        if layout is None:
            raise ValueError("Engram requires at least one configured layer")
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                "Engram compressed vocabulary mismatch: "
                f"checkpoint={config.engram_compressed_vocab_size}, tokenizer={vocab_size}"
            )

        primes = torch.tensor(layout.primes, dtype=torch.int64)
        sizes = primes.flatten(1)
        self.register_buffer(
            "token_map",
            torch.tensor(token_map, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", sizes.cumsum(-1) - sizes, persistent=False)
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size),
            persistent=False,
        )
        self.max_ngram_size = layout.max_ngram_size
        self.lookback_depth = layout.max_ngram_size - 1
        self.required_num_embeddings = tuple(sum(size for ngram in layer for size in ngram) for layer in layout.primes)
        self.pad_id = token_map[config.engram_pad_token_id]
        self.image_token_id = config.image_token_id
        self.image_pad_token_id = getattr(
            config,
            "image_pad_token_id",
            self.image_token_id + 1,
        )
        logger.debug(
            "FOR-ENGRAM hash state initialized: layers=%s max_ngram_size=%d "
            "heads=%d tokenizer_vocab=%d compressed_vocab=%d",
            layout.layer_ids,
            self.max_ngram_size,
            layout.n_heads,
            len(token_map),
            vocab_size,
        )

    def _dead_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        valid_ids = (token_ids >= 0) & (token_ids < self.token_map.numel())
        text = valid_engram_token_mask(
            token_ids,
            self.image_token_id,
            self.image_pad_token_id,
        )
        return ~valid_ids | ~text

    def _compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Clamp sentinels before indexing. Missing history and image tokens are
        # replaced with pad below, after their barrier semantics are recorded.
        safe_ids = token_ids.clamp(0, self.token_map.numel() - 1)
        return self.token_map[safe_ids]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        lookback_dead_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[tokens, layers, hash_columns]`` IDs and active-token mask."""
        input_ids = input_ids.long()
        positions = positions.long()
        query_start_loc = query_start_loc.to(device=input_ids.device, dtype=torch.long)
        num_tokens = input_ids.numel()
        num_requests = query_start_loc.numel() - 1
        columns = (self.max_ngram_size - 1) * self.primes.shape[-1]
        if num_tokens == 0:
            return (
                input_ids.new_empty((0, self.primes.shape[0], columns)),
                input_ids.new_empty((0,), dtype=torch.bool),
            )
        if positions.shape != input_ids.shape or num_requests < 1:
            raise ValueError("Engram positions and query_start_loc do not match input_ids")

        if lookback_token_ids is not None and tuple(lookback_token_ids.shape) != (num_requests, self.lookback_depth):
            raise ValueError("Engram lookback_token_ids shape mismatch")
        if lookback_dead_mask is not None and tuple(lookback_dead_mask.shape) != (
            num_requests,
            self.lookback_depth,
        ):
            raise ValueError("Engram lookback_dead_mask shape does not match lookback_token_ids")
        if dead_mask is not None and dead_mask.shape != input_ids.shape:
            raise ValueError("Engram dead_mask must have one entry per input token")

        validated_current_dead = self._dead_mask(input_ids)
        if dead_mask is not None:
            validated_current_dead |= dead_mask.to(
                device=input_ids.device, dtype=torch.bool
            )

        if lookback_token_ids is None:
            lookback_token_ids = input_ids.new_full(
                (num_requests, self.lookback_depth), self.DEAD
            )
            lookback_dead_mask = torch.ones(
                (num_requests, self.lookback_depth),
                dtype=torch.bool,
                device=input_ids.device,
            )
        else:
            lookback_token_ids = lookback_token_ids.to(
                device=input_ids.device, dtype=torch.long
            )
            validated_history_dead = self._dead_mask(lookback_token_ids)
            if lookback_dead_mask is not None:
                validated_history_dead |= lookback_dead_mask.to(
                    device=input_ids.device, dtype=torch.bool
                )
            lookback_dead_mask = validated_history_dead

        request_ids = torch.repeat_interleave(
            torch.arange(num_requests, device=input_ids.device),
            query_start_loc[1:] - query_start_loc[:-1],
            output_size=num_tokens,
        )
        if request_ids.numel() != num_tokens:
            raise ValueError("Engram query_start_loc does not cover input_ids")
        local_positions = torch.arange(num_tokens, device=input_ids.device) - query_start_loc[request_ids]

        return self.forward_packed(
            input_ids,
            positions,
            request_ids,
            local_positions,
            lookback_token_ids,
            lookback_dead_mask,
            validated_current_dead,
        )

    def forward_packed(
        self,
        input_ids,
        positions,
        request_ids,
        local_positions,
        lookback_token_ids,
        lookback_dead_mask,
        dead_mask,
        single_token_per_request=False,
    ):
        """Hash runner-provided device mappings without data-dependent shapes.

        All inputs have static capacity or the graph's token dimension. Padding
        has request/local index zero and dead_mask=True. No host metadata read,
        H2D query-boundary copy, or repeat_interleave is needed during replay.
        """
        input_ids = input_ids.long()
        positions = positions.long()
        request_ids = request_ids[: input_ids.numel()]
        local_positions = local_positions[: input_ids.numel()]
        dead_mask = dead_mask[: input_ids.numel()]
        num_tokens = input_ids.numel()
        if positions.shape != input_ids.shape:
            raise ValueError("Engram positions do not match input_ids")
        if request_ids.shape != input_ids.shape or local_positions.shape != input_ids.shape:
            raise ValueError("Engram device mappings do not cover input_ids")
        if dead_mask.shape != input_ids.shape:
            raise ValueError("Engram dead_mask must have one entry per input token")
        num_requests = lookback_token_ids.shape[0]
        current_dead = (
            dead_mask.to(device=input_ids.device, dtype=torch.bool)
            | (input_ids == self.image_token_id)
            | (input_ids == self.image_pad_token_id)
        )
        lookback_token_ids = lookback_token_ids.to(device=input_ids.device, dtype=torch.long)
        expected_lookback_shape = (num_requests, self.lookback_depth)
        if tuple(lookback_token_ids.shape) != expected_lookback_shape:
            raise ValueError(
                "Engram lookback_token_ids shape mismatch: "
                f"got {tuple(lookback_token_ids.shape)}, expected {expected_lookback_shape}"
            )
        if lookback_dead_mask is None:
            raise ValueError("Engram packed hash requires lookback_dead_mask")
        if tuple(lookback_dead_mask.shape) != expected_lookback_shape:
            raise ValueError("Engram lookback_dead_mask shape does not match lookback_token_ids")
        # The runner stages token IDs and their authoritative dead mask
        # together. Revalidating history here adds comparisons to the side
        # stream before the first fetch without strengthening that contract.
        history_dead = lookback_dead_mask.to(device=input_ids.device, dtype=torch.bool)

        compressed_current = self._compress(input_ids)

        if single_token_per_request:
            if num_tokens > num_requests:
                raise ValueError(
                    "Engram single-token path requires one history row per padded token"
                )
            # Decode packs one token per request in request-row order. Reuse
            # that layout directly, as recipes does with its dense [B, Q]
            # decode input, instead of gathering every n-gram source again.
            compressed_history = self._compress(lookback_token_ids[:num_tokens])
            encoded_history = torch.where(
                history_dead[:num_tokens], self.DEAD, compressed_history
            )
            source_columns = encoded_history.unbind(dim=-1)
        else:
            compressed_history = self._compress(lookback_token_ids)
            encoded_current = torch.where(
                current_dead, self.DEAD, compressed_current
            )
            encoded_history = torch.where(
                history_dead, self.DEAD, compressed_history
            )
            token_rows = torch.arange(
                num_tokens,
                dtype=local_positions.dtype,
                device=input_ids.device,
            )
            source_columns = []
            for shift in range(1, self.max_ngram_size):
                from_current = local_positions >= shift
                current_indices = (token_rows - shift).clamp_min(0)
                history_indices = (shift - local_positions - 1).clamp(
                    0, self.lookback_depth - 1
                )
                source_columns.append(
                    torch.where(
                        from_current,
                        encoded_current[current_indices],
                        encoded_history[request_ids, history_indices],
                    )
                )

        blocked = current_dead | (positions < 0)
        window_columns = [torch.where(blocked, self.pad_id, compressed_current)]
        for shift, source in enumerate(source_columns, start=1):
            blocked = blocked | (source == self.DEAD) | (positions < shift)
            window_columns.append(
                torch.where(blocked, self.pad_id, source)
            )

        window = torch.stack(window_columns, dim=-1)

        products = window[:, None, :] * self.multipliers[None, :, :]
        rolling = products[..., 0]
        hashes = []
        for shift in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, shift - 1])
        return torch.cat(hashes, dim=-1) + self.offsets, ~current_dead
