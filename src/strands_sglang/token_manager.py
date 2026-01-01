"""Token management for TITO (Token-In/Token-Out) training.

This module provides:
- Token: A single token with ID, text, and optional logprob
- TokenManager: Manages segment-based token accumulation with INPUT/OUTPUT tracking

For RL training, you typically want:
- token_ids: Flat list of all tokens for the trajectory
- output_mask: Boolean mask for loss computation (True = model output)
- logprobs: Log probabilities for policy gradient
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class Token:
    """A single token with its ID, text, and optional logprob.

    Logprob is available when:
    - OUTPUT segments: Always (from model generation)
    - INPUT segments: When return_logprobs=True (computed by forward pass)
    """

    token_id: int
    text: str = ""
    logprob: float | None = None


class TokenManager:
    """Manages token accumulation with segment-based INPUT/OUTPUT tracking.

    Tokens are organized into segments, where each segment is either:
    - INPUT: Tokenized prompts, tool results
    - OUTPUT: Model generations

    Both segment types can have logprobs when return_logprobs=True.

    Example (without input logprobs):
        >>> manager = TokenManager(tokenizer)
        >>> manager.add_input([1, 2, 3])
        >>> manager.add_output([4, 5], [0.1, 0.2])
        >>> manager.logprobs         # [None, None, None, 0.1, 0.2]

    Example (with input logprobs for distillation):
        >>> manager = TokenManager(tokenizer)
        >>> manager.add_input([1, 2, 3], [-0.5, -0.3, -0.1])
        >>> manager.add_output([4, 5], [0.1, 0.2])
        >>> manager.logprobs         # [-0.5, -0.3, -0.1, 0.1, 0.2]  # No None!
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Create a TokenManager.

        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens.
        """
        self.tokenizer = tokenizer
        self._segments: list[list[int]] = []
        self._segment_logprobs: list[list[float] | None] = []
        self._segment_is_output: list[bool] = []

    def reset(self) -> None:
        """Reset token accumulation for a new episode."""
        self._segments = []
        self._segment_logprobs = []
        self._segment_is_output = []

    def add_input(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add an INPUT segment (tokenized prompts, tool results).

        Args:
            token_ids: Token IDs for this segment.
            logprobs: Optional log probabilities (from forward pass when return_logprobs=True).
        """
        if token_ids:
            self._segments.append(list(token_ids))
            self._segment_logprobs.append(logprobs)
            self._segment_is_output.append(False)

    def add_output(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add an OUTPUT segment (model generation).

        OUTPUT segments may have logprobs if the model provides them.

        Args:
            token_ids: Token IDs for this segment.
            logprobs: Optional log probabilities for each token.
        """
        if token_ids:
            self._segments.append(list(token_ids))
            self._segment_logprobs.append(logprobs)
            self._segment_is_output.append(True)

    @property
    def token_ids(self) -> list[int]:
        """Get all token IDs as a flat list."""
        return [tok for segment in self._segments for tok in segment]

    @property
    def segments(self) -> list[list[int]]:
        """Get token IDs organized by segment.

        Returns:
            List of token ID lists, one per segment.
        """
        return [list(seg) for seg in self._segments]

    @property
    def output_mask(self) -> list[bool]:
        """Get a boolean mask indicating which tokens are model outputs.

        Use this for loss computation in RL training - only compute loss
        on tokens where mask is True (model outputs).

        Returns:
            List of booleans, same length as token_ids.
            True = model output (compute loss), False = input (no loss).
        """
        mask: list[bool] = []
        for segment, is_output in zip(self._segments, self._segment_is_output):
            mask.extend([is_output] * len(segment))
        return mask

    @property
    def logprobs(self) -> list[float | None]:
        """Get log probabilities for all tokens (flat list).

        Returns None for tokens without logprobs, float otherwise.
        When return_logprobs=True, all tokens will have logprobs (no None).

        Returns:
            List of logprobs, same length as token_ids.
        """
        result: list[float | None] = []
        for token_ids, logprobs in zip(self._segments, self._segment_logprobs):
            if logprobs is not None:
                result.extend(logprobs)
            else:
                result.extend([None] * len(token_ids))
        return result

    @property
    def segment_info(self) -> list[tuple[bool, int]]:
        """Get segment metadata (is_output, length) for each segment.

        Returns:
            List of (is_output, segment_length) tuples.
        """
        return [(is_out, len(seg)) for is_out, seg in zip(self._segment_is_output, self._segments)]

    def get_tokens(self, decode: bool = True) -> list[list[Token]]:
        """Get tokens organized by segment as Token objects.

        Args:
            decode: Whether to decode token IDs to text. Default True.

        Returns:
            List of Token lists, one per segment.
        """
        result: list[list[Token]] = []

        for token_ids, logprobs in zip(self._segments, self._segment_logprobs):
            segment_tokens: list[Token] = []

            for tok_idx, token_id in enumerate(token_ids):
                # Get logprob if available
                logprob: float | None = None
                if logprobs is not None and tok_idx < len(logprobs):
                    logprob = logprobs[tok_idx]

                # Decode token text if requested
                text = ""
                if decode:
                    text = self.tokenizer.decode([token_id])

                segment_tokens.append(Token(token_id=token_id, text=text, logprob=logprob))

            result.append(segment_tokens)

        return result

    def __len__(self) -> int:
        """Return total number of tokens."""
        return sum(len(seg) for seg in self._segments)

    def __repr__(self) -> str:
        """Return string representation."""
        n_segments = len(self._segments)
        n_tokens = len(self)
        n_output = sum(len(seg) for seg, is_out in zip(self._segments, self._segment_is_output) if is_out)
        return f"TokenManager(segments={n_segments}, tokens={n_tokens}, output_tokens={n_output})"
