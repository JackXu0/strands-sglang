"""SGLang native /generate API model provider for token-in/token-out training.

This provider uses SGLang's native HTTP APIs:
- `/generate` for text generation (returns output_ids directly)

It uses a HuggingFace tokenizer for:
- Applying chat templates (via tokenizer.apply_chat_template())
- Tokenizing prompts and tool results

This eliminates retokenization drift in RL training by maintaining token IDs
throughout the rollout instead of converting text back to tokens.
"""

from __future__ import annotations

import json
import logging
import re
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterable,
    Type,
    TypedDict,
    cast,
)

import httpx
from pydantic import BaseModel
from strands.models import Model
from strands.models.openai import OpenAIModel
from strands.types.content import Messages, SystemContentBlock
from strands.types.event_loop import Metrics, Usage
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import Unpack, override

from .token_manager import TokenManager
from .tool_parser import PARSE_ERROR_TOOL_NAME, HermesToolCallParser, ToolCallParser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class SGLangModel(Model):
    """SGLang native `/generate` API provider with token-in/token-out support.

    This model provider:
    - Uses a HuggingFace tokenizer for chat template formatting and tokenization
    - Calls SGLang's `/generate` endpoint which returns output_ids directly
    - Tracks accumulated token_ids via TokenManager for training

    Attributes:
        token_manager: TokenManager for accessing accumulated tokens, logprobs, and masks.
        tokenizer: The HuggingFace tokenizer used for encoding/decoding.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct")
        >>> model = SGLangModel(tokenizer=tokenizer, base_url="http://localhost:8000")
        >>> # After agent step:
        >>> model.token_manager.token_ids      # Full token trajectory
        >>> model.token_manager.output_mask    # Boolean mask for loss computation
        >>> model.token_manager.logprobs       # Log probabilities (all tokens if return_logprobs=True)
    """

    class SGLangConfig(TypedDict, total=False):
        """Configuration options for SGLang native API."""

        base_url: str
        model_id: str | None
        params: dict[str, Any] | None  # default sampling params
        timeout: float | tuple[float, float] | None
        tool_call_parser: ToolCallParser | None  # custom parser for tool calls
        return_logprobs: bool  # return logprobs for all tokens

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        **model_config: Unpack[SGLangConfig],
    ) -> None:
        """Create an SGLang model client.

        Args:
            tokenizer: HuggingFace tokenizer for chat template and tokenization.
            **model_config: Configuration options including:
                - base_url: SGLang server URL (default: http://localhost:8000)
                - model_id: Optional model identifier
                - params: Default sampling parameters
                - timeout: Request timeout in seconds
                - return_logprobs: Compute logprobs for all tokens. Default True.
        """
        self.tokenizer = tokenizer

        base_url = str(model_config.get("base_url") or "http://localhost:8000").rstrip("/")
        timeout = model_config.get("timeout")
        if isinstance(timeout, tuple):
            timeout_obj = httpx.Timeout(connect=timeout[0], read=timeout[1])
        else:
            timeout_obj = httpx.Timeout(timeout or 300.0)

        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout_obj)
        self.config = dict(model_config)
        self.config["base_url"] = base_url

        # Token manager for TITO tracking
        self.token_manager = TokenManager(tokenizer)
        self._processed_message_count: int = 0
        self._current_tools: list[dict] | None = None

        # Tool call parser (defaults to Hermes/Qwen XML format)
        # For RL training: uses strict parsing with no post-processing
        self._tool_call_parser: ToolCallParser = model_config.get("tool_call_parser") or HermesToolCallParser()

        logger.debug("config=<%s> | initializing", self.config)

    def reset(self) -> None:
        """Reset token accumulation for a new episode.

        Call this at episode start. Clears all accumulated tokens and resets
        internal state for tool tracking.
        """
        self.token_manager.reset()
        self._processed_message_count = 0
        self._current_tools = None

    # -------------------------------------------------------------------------
    # Model interface implementation
    # -------------------------------------------------------------------------

    @override
    def update_config(self, **model_config: Unpack[SGLangConfig]) -> None:  # type: ignore[override]
        """Update the model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        if "base_url" in model_config and model_config["base_url"]:
            self.config["base_url"] = str(model_config["base_url"]).rstrip("/")
        self.config.update(model_config)

    @override
    def get_config(self) -> SGLangConfig:
        """Get the model configuration.

        Returns:
            The model configuration dict.
        """
        return cast(SGLangModel.SGLangConfig, self.config)

    # -------------------------------------------------------------------------
    # Chat template and message formatting
    # -------------------------------------------------------------------------

    @staticmethod
    def messages_to_openai(messages: Messages, system_prompt: str | None = None) -> list[dict[str, Any]]:
        """Convert strands Messages to OpenAI format for chat templates.

        Uses strands' OpenAIModel formatter and flattens content
        for compatibility with HuggingFace apply_chat_template.
        """

        result = OpenAIModel.format_request_messages(messages=messages, system_prompt=system_prompt)

        # Flatten content from [{"type": "text", "text": "..."}] to "..."
        for message in result:
            if "content" in message and isinstance(message["content"], list):
                if len(message["content"]) > 0 and "text" in message["content"][0]:
                    message["content"] = message["content"][0]["text"]
                else:
                    message["content"] = ""

            # When tool_calls exist, strip <tool_call> markup from content to avoid duplication.
            # Strands stores both the raw <tool_call> text AND structured tool_calls data.
            # apply_chat_template would render both, causing duplicate tool calls.
            if message.get("tool_calls") and message.get("content"):
                # Remove <tool_call>...</tool_call> blocks from content
                message["content"] = re.sub(
                    r"<tool_call>.*?</tool_call>\s*",
                    "",
                    message["content"],
                    flags=re.DOTALL,
                ).strip()

        return result

    def _convert_tool_specs(self, tool_specs: list[ToolSpec] | None) -> list[dict] | None:
        """Convert strands ToolSpec to tokenizer format (OpenAI-like).

        Args:
            tool_specs: List of strands tool specifications.

        Returns:
            List of tools in OpenAI format for tokenizer, or None.
        """
        if not tool_specs:
            return None

        tools = []
        for spec in tool_specs:
            tool = {
                "type": "function",
                "function": {
                    "name": spec.get("name", ""),
                    "description": spec.get("description", ""),
                    "parameters": spec.get("inputSchema", {}),
                },
            }
            tools.append(tool)
        return tools

    def _apply_chat_template(
        self,
        messages: Messages,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """Apply chat template to messages.

        Args:
            messages: Strands messages to format.
            system_prompt: System prompt.
            tools: Tools in OpenAI format for tokenizer.

        Returns:
            Formatted prompt string with chat template applied.
        """
        chat_messages = self.messages_to_openai(messages, system_prompt)

        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools

        return self.tokenizer.apply_chat_template(chat_messages, **kwargs)

    def _format_continuation(self, new_messages: Messages) -> str:
        """Format new messages for continuation (e.g., tool results).

        Note: Tools are NOT included here since they're already in the context
        from the first call. Including them again would cause duplication.

        Args:
            new_messages: New messages to add to context.

        Returns:
            Formatted string for the new messages.
        """
        chat_messages = self.messages_to_openai(new_messages)

        return self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _parse_logprobs(
        self,
        raw_logprobs: list,
        expected_token_ids: list[int] | None = None,
    ) -> list[float]:
        """Parse SGLang logprobs format and optionally verify token IDs.

        SGLang returns logprobs as: [[logprob, token_id, top_logprobs], ...]
        - logprob: float, the log probability
        - token_id: int, the token ID (for sanity checking)
        - top_logprobs: null or list of alternative tokens

        Args:
            raw_logprobs: Raw logprobs from SGLang response.
            expected_token_ids: Optional token IDs to verify against.

        Returns:
            List of logprob floats.

        Raises:
            ValueError: If token IDs don't match (sanity check failure).
        """
        logprobs: list[float] = []

        for i, entry in enumerate(raw_logprobs):
            if not isinstance(entry, list) or len(entry) < 2:
                logprobs.append(0.0)
                continue

            lp, token_id = entry[0], entry[1]

            # Sanity check: verify token_id matches expected
            if expected_token_ids is not None and i < len(expected_token_ids):
                if token_id != expected_token_ids[i]:
                    logger.warning(
                        "Logprob token_id mismatch at index %d: expected %d, got %d", i, expected_token_ids[i], token_id
                    )

            logprobs.append(float(lp) if lp is not None else 0.0)

        return logprobs

    def _build_generate_payload(
        self,
        *,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        stream: bool,
        return_logprob: bool = False,
    ) -> dict[str, Any]:
        """Build the request payload for /generate.

        Args:
            input_ids: Tokenized input.
            sampling_params: Sampling parameters.
            stream: Whether to stream the response.
            return_logprob: Whether to return logprobs (top-level param, not in sampling_params).

        Returns:
            Request payload dict.
        """
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "stream": stream,
        }

        model_id = self.get_config().get("model_id")
        if model_id:
            payload["model"] = model_id

        if sampling_params:
            payload["sampling_params"] = sampling_params

        if return_logprob:
            payload["return_logprob"] = True
            payload["logprob_start_len"] = 0  # Return input logprobs from position 0

        return payload

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream generation from SGLang /generate endpoint.

        This method:
        1. On first call: applies chat template and tokenizes full prompt
        2. On subsequent calls: tokenizes only NEW messages (tool results)
        3. Calls /generate with accumulated input_ids
        4. Appends output_ids to step token accumulator

        Args:
            messages: Conversation messages.
            tool_specs: Tool specifications (converted and included in chat template).
            system_prompt: System prompt text.
            tool_choice: Ignored (tool choice is implicit in the prompt).
            system_prompt_content: Ignored (use system_prompt instead).
            **kwargs: Additional options:
                - sampling_params: Override default sampling params

        Yields:
            StreamEvent dicts for strands Agent compatibility.

        Raises:
            ContextWindowOverflowException: If context too long.
            ModelThrottledException: If rate limited.
        """
        # Convert tool_specs to tokenizer format and store for continuations
        if tool_specs and not self._current_tools:
            self._current_tools = self._convert_tool_specs(tool_specs)
            logger.debug("tool_specs converted: %d tools", len(self._current_tools))

        # Build sampling params from config
        sampling_params: dict[str, Any] = dict(self.get_config().get("params") or {})

        # Whether to request logprobs (for TITO training)
        return_logprobs = self.get_config().get("return_logprobs", True)

        # Determine new input tokens (we'll add them AFTER response to get logprobs)
        current_message_count = len(messages)
        new_input_tokens: list[int] | None = None

        if len(self.token_manager) == 0:
            # First call: apply chat template and tokenize full prompt
            formatted = self._apply_chat_template(messages, system_prompt, tools=self._current_tools)
            new_input_tokens = self.tokenizer.encode(formatted, add_special_tokens=False)
        else:
            # Subsequent call: tokenize only NEW messages (no tools - already in context)
            if current_message_count > self._processed_message_count:
                new_messages = messages[self._processed_message_count :]
                new_formatted = self._format_continuation(new_messages)
                new_input_tokens = self.tokenizer.encode(new_formatted, add_special_tokens=False)

        # Build input_ids: existing tokens + new input tokens
        if new_input_tokens:
            input_ids = self.token_manager.token_ids + new_input_tokens
        else:
            input_ids = self.token_manager.token_ids

        # Build request payload
        payload = self._build_generate_payload(
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=True,
            return_logprob=return_logprobs,
        )

        # Yield message start events
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}

        # Track state during streaming
        prev_text = ""
        last_output_ids: list[int] = []
        last_output_logprobs: list[float] | None = None
        last_input_logprobs: list[float] | None = None
        last_meta: dict[str, Any] | None = None

        try:
            async with self.client.stream("POST", "/generate", json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data_content = line[len("data:") :].strip()
                    if data_content == "[DONE]":
                        break

                    try:
                        event = json.loads(data_content)
                    except json.JSONDecodeError:
                        continue

                    # Extract text delta
                    new_text = event.get("text")
                    if isinstance(new_text, str):
                        # SGLang returns cumulative text, extract delta
                        if new_text.startswith(prev_text):
                            delta = new_text[len(prev_text) :]
                        else:
                            delta = new_text
                        prev_text = new_text
                        if delta:
                            yield {"contentBlockDelta": {"delta": {"text": delta}}}

                    # Capture output_ids
                    output_ids = event.get("output_ids")
                    if isinstance(output_ids, list) and all(isinstance(x, int) for x in output_ids):
                        last_output_ids = cast(list[int], output_ids)

                    # Capture logprobs if provided by SGLang
                    # SGLang returns logprobs as [[logprob, token_id, ?], ...]
                    meta_info = event.get("meta_info", {})

                    output_logprobs = meta_info.get("output_token_logprobs")
                    if output_logprobs is None:
                        output_logprobs = event.get("output_token_logprobs")
                    if isinstance(output_logprobs, list) and output_logprobs:
                        # Parse with sanity check against output_ids
                        last_output_logprobs = self._parse_logprobs(
                            output_logprobs,
                            expected_token_ids=last_output_ids if last_output_ids else None,
                        )

                    # Capture input logprobs
                    # Note: logprob_start_len=0 ensures input logprobs are aligned with input_ids
                    input_logprobs = meta_info.get("input_token_logprobs")
                    if input_logprobs is None:
                        input_logprobs = event.get("input_token_logprobs")
                    if isinstance(input_logprobs, list) and input_logprobs:
                        # Parse with sanity check against input_ids
                        last_input_logprobs = self._parse_logprobs(
                            input_logprobs,
                            expected_token_ids=input_ids,
                        )

                    # Capture metadata (reuse meta_info from above)
                    if meta_info:
                        last_meta = meta_info

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 400:
                raise ContextWindowOverflowException(str(e)) from e
            if status in (429, 503):
                raise ModelThrottledException(str(e)) from e
            raise

        # Add input tokens with logprobs (if any)
        if new_input_tokens:
            new_input_logprobs: list[float] | None = None
            if return_logprobs and last_input_logprobs:
                # Extract logprobs for just the NEW input tokens
                # input_token_logprobs covers entire prompt, we want the last N
                new_input_logprobs = last_input_logprobs[-len(new_input_tokens) :]
            self.token_manager.add_input(new_input_tokens, new_input_logprobs)

        # Add output tokens with logprobs
        if last_output_ids:
            self.token_manager.add_output(last_output_ids, last_output_logprobs)

        # Update message count (+1 for new assistant message)
        self._processed_message_count = current_message_count + 1

        # Parse tool calls with strict validation (no post-processing for RL training)
        parsed_tool_calls = self._tool_call_parser.parse(prev_text)

        # Yield content block stop for text block
        yield {"contentBlockStop": {}}

        # Yield successful tool calls as toolUse blocks
        for tool_call in parsed_tool_calls.tool_calls:
            # Start toolUse block
            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": tool_call.name,
                        }
                    }
                }
            }
            # Delta with input
            yield {
                "contentBlockDelta": {
                    "delta": {
                        "toolUse": {
                            "input": json.dumps(tool_call.input),
                        }
                    }
                }
            }
            # Stop toolUse block
            yield {"contentBlockStop": {}}

        # Yield parse errors as synthetic tool calls so model gets error feedback
        # This is critical for RL training - model learns from seeing what went wrong
        for error in parsed_tool_calls.errors:
            # Use original tool name if we could extract it, otherwise fall back to error tool
            tool_name = error.attempted_name or PARSE_ERROR_TOOL_NAME
            logger.warning(
                "Yielding parse error for tool '%s': %s",
                tool_name,
                error.error_message[:100],
            )
            # Start toolUse block - use original tool name so error flows naturally
            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": error.id,
                            "name": tool_name,
                        }
                    }
                }
            }
            # Delta with error details as the malformed input
            # The tool executor will fail validation and return proper error
            yield {
                "contentBlockDelta": {
                    "delta": {
                        "toolUse": {
                            "input": error.raw_content[:2000],  # Pass raw content, let tool validate
                        }
                    }
                }
            }
            # Stop toolUse block
            yield {"contentBlockStop": {}}

        # Determine stop reason
        # Note: has_tool_calls is True if ANY tool call was attempted (success or error)
        # This ensures the agent continues the loop even for malformed tool calls
        stop_reason: str = "tool_use" if parsed_tool_calls.has_tool_calls else "end_turn"
        if last_meta and isinstance(last_meta.get("finish_reason"), dict):
            fr = cast(dict[str, Any], last_meta.get("finish_reason"))
            if fr.get("type") == "length":
                stop_reason = "max_tokens"

        yield {"messageStop": {"stopReason": cast(Any, stop_reason)}}

        # Yield metadata with usage info
        if last_meta:
            usage: Usage = {
                "inputTokens": int(last_meta.get("prompt_tokens") or 0),
                "outputTokens": int(last_meta.get("completion_tokens") or 0),
                "totalTokens": int((last_meta.get("prompt_tokens") or 0) + (last_meta.get("completion_tokens") or 0)),
            }
            latency_ms = int(float(last_meta.get("e2e_latency") or 0.0) * 1000)
            metrics: Metrics = {"latencyMs": latency_ms}
            yield {"metadata": {"usage": usage, "metrics": metrics}}

    @override
    async def structured_output(
        self,
        output_model: Type[BaseModel],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, BaseModel | Any], None]:
        """Not implemented for training-only model.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("structured_output is not supported by SGLangModel (training-only)")
        # Make this a generator to satisfy the type signature
        yield {}  # pragma: no cover
