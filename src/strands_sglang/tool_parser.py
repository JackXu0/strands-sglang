"""Tool call parsing for model outputs with strict error handling for RL training.

Different models/tokenizers use different formats for tool calls in their chat templates.
This module provides parsers that return both successful parses AND errors,
enabling models to learn from malformed outputs during training.

Currently supported:
- HermesToolCallParser: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  (Used by Qwen, Hermes, and many other instruction-tuned models)

Design for RL Training:
- NO post-processing of model outputs (strict parsing)
- Native JSON error messages returned directly
- Parse errors become synthetic tool calls that return error feedback
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Regex to extract tool name even from malformed JSON
_NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _make_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


# Reserved tool name for parse errors - allows error feedback to flow to model
PARSE_ERROR_TOOL_NAME = "__tool_call_parse_error__"


@dataclass
class ToolCall:
    """A successfully parsed tool call."""

    id: str
    name: str
    input: dict


@dataclass
class ToolCallError:
    """A failed tool call parse attempt with detailed error info."""

    id: str
    raw_content: str
    error_message: str
    attempted_name: str | None = None  # Name if we could extract it before failure


@dataclass
class ParseResult:
    """Result of parsing tool calls from model output.

    For RL training, both successful parses and errors are returned so that:
    - Successful tool calls are executed normally
    - Parse errors become visible to the model as error feedback
    """

    tool_calls: list[ToolCall]
    errors: list[ToolCallError]

    @property
    def has_errors(self) -> bool:
        """Check if there were any parse errors."""
        return len(self.errors) > 0

    @property
    def has_tool_calls(self) -> bool:
        """Check if any tool calls were attempted (successful or not)."""
        return len(self.tool_calls) > 0 or len(self.errors) > 0


class ToolCallParser(ABC):
    """Base class for tool call parsers.

    Subclasses should implement the `parse` method to extract tool calls
    from model output text with strict validation and detailed error reporting.

    Example:
        >>> parser = HermesToolCallParser()
        >>> result = parser.parse('<tool_call>{"name": "foo", "arguments": {}}</tool_call>')
        >>> print(result.tool_calls[0].name)
        foo
    """

    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """Parse tool calls from model output text.

        Args:
            text: Model output text.

        Returns:
            ParseResult containing successful tool calls and any parse errors.
        """
        ...

    def __call__(self, text: str) -> list[dict]:
        """Parse tool calls (callable interface for backwards compatibility).

        Note: This returns only successful parses. For RL training,
        use parse() directly to also get errors.

        Args:
            text: Model output text.

        Returns:
            List of parsed tool calls as dicts.
        """
        result = self.parse(text)
        return [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in result.tool_calls]


class HermesToolCallParser(ToolCallParser):
    """Parser for Hermes/Qwen XML tool call format with strict validation.

    Format: <tool_call>{"name": "func", "arguments": {"arg": value}}</tool_call>

    Used by:
    - Qwen/Qwen2/Qwen3 models
    - NousResearch/Hermes models
    - Models using similar XML-wrapped JSON tool call format

    This parser is STRICT for RL training:
    - No post-processing or fixing of malformed JSON
    - Native JSON error messages returned directly
    - Parse errors are returned alongside successful parses
    """

    BOT_TOKEN = "<tool_call>"
    EOT_TOKEN = "</tool_call>"

    _PATTERN = re.compile(
        rf"{re.escape(BOT_TOKEN)}\s*(.*?)\s*{re.escape(EOT_TOKEN)}",
        re.DOTALL,
    )

    def parse(self, text: str) -> ParseResult:
        """Parse tool calls with strict validation.

        Args:
            text: Model output text.

        Returns:
            ParseResult with successful tool calls and detailed errors.
        """
        tool_calls: list[ToolCall] = []
        errors: list[ToolCallError] = []

        for match in self._PATTERN.finditer(text):
            raw_content = match.group(1).strip()
            tool_call_id = _make_tool_call_id()

            # Attempt strict JSON parse - NO fixing or post-processing
            try:
                call_json = json.loads(raw_content)
            except json.JSONDecodeError as e:
                # Try to extract tool name with regex even if JSON is malformed
                name_match = _NAME_PATTERN.search(raw_content)
                attempted_name = name_match.group(1) if name_match else None

                error_msg = f"JSON parse error: {e}"
                errors.append(
                    ToolCallError(
                        id=tool_call_id,
                        raw_content=raw_content,
                        error_message=error_msg,
                        attempted_name=attempted_name,
                    )
                )
                logger.warning("Tool call parse error: %s", error_msg)
                continue

            # Validate required fields
            if not isinstance(call_json, dict):
                errors.append(
                    ToolCallError(
                        id=tool_call_id,
                        raw_content=raw_content,
                        error_message="Tool call must be a JSON object, not " + type(call_json).__name__,
                        attempted_name=None,
                    )
                )
                continue

            name = call_json.get("name")
            if not name:
                errors.append(
                    ToolCallError(
                        id=tool_call_id,
                        raw_content=raw_content,
                        error_message=(
                            "Tool call missing required field 'name'. "
                            "Expected format: {\"name\": \"tool_name\", \"arguments\": {...}}"
                        ),
                        attempted_name=None,
                    )
                )
                continue

            if not isinstance(name, str):
                errors.append(
                    ToolCallError(
                        id=tool_call_id,
                        raw_content=raw_content,
                        error_message=f"Tool call 'name' must be a string, got {type(name).__name__}",
                        attempted_name=str(name) if name else None,
                    )
                )
                continue

            # Extract arguments (optional, defaults to empty dict)
            arguments = call_json.get("arguments", {})
            if not isinstance(arguments, dict):
                errors.append(
                    ToolCallError(
                        id=tool_call_id,
                        raw_content=raw_content,
                        error_message=f"Tool call 'arguments' must be an object, got {type(arguments).__name__}",
                        attempted_name=name,
                    )
                )
                continue

            # Success!
            tool_calls.append(
                ToolCall(
                    id=tool_call_id,
                    name=name,
                    input=arguments,
                )
            )

        return ParseResult(tool_calls=tool_calls, errors=errors)
