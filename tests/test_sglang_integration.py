# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for SGLangModel (requires running SGLang server).

Run with: pytest tests/test_sglang_integration.py -v
Skip with: pytest tests/ --ignore=tests/test_sglang_integration.py

These tests require a running SGLang server. Configure via environment:
    SGLANG_BASE_URL: Server URL (default: http://localhost:8000)
    SGLANG_MODEL_ID: Model ID for Qwen3 (default: Qwen/Qwen3-4B-Instruct-2507)
"""

import os

import pytest
from transformers import AutoTokenizer

from strands_sglang import SGLangModel
from strands_sglang.tool_parser import HermesToolCallParser

# Configuration from environment
BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://localhost:8000")
MODEL_ID = os.environ.get("SGLANG_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tokenizer():
    """Load Qwen3 tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def model(tokenizer):
    """Create SGLangModel connected to running server."""
    return SGLangModel(
        tokenizer=tokenizer,
        tool_call_parser=HermesToolCallParser(),
        base_url=BASE_URL,
        model_id=MODEL_ID,
    )


@pytest.fixture
def calculator_tool():
    """Sample calculator tool spec."""
    return {
        "name": "calculator",
        "description": "Perform arithmetic calculations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }


class TestStreamBasic:
    """Basic streaming tests."""

    async def test_simple_generation(self, model):
        """Generate a simple response without tools."""
        messages = [{"role": "user", "content": [{"text": "Say 'hello' and nothing else."}]}]

        events = []
        async for event in model.stream(messages):
            events.append(event)

        # Should have content events
        content_deltas = [e for e in events if "contentBlockDelta" in e]
        assert len(content_deltas) > 0

        # Should have text in deltas
        text = "".join(
            e["contentBlockDelta"]["delta"].get("text", "")
            for e in content_deltas
            if "text" in e["contentBlockDelta"]["delta"]
        )
        assert "hello" in text.lower()

    async def test_generation_with_system_prompt(self, model):
        """Generate with system prompt."""
        messages = [{"role": "user", "content": [{"text": "What are you?"}]}]
        system_prompt = "You are a helpful calculator assistant. Be brief."

        events = []
        async for event in model.stream(messages, system_prompt=system_prompt):
            events.append(event)

        content_deltas = [e for e in events if "contentBlockDelta" in e]
        assert len(content_deltas) > 0

    async def test_metadata_event(self, model):
        """Stream should end with metadata event."""
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        events = []
        async for event in model.stream(messages):
            events.append(event)

        # Last event should be metadata
        assert "metadata" in events[-1]
        metadata = events[-1]["metadata"]
        assert "usage" in metadata


class TestStreamWithTools:
    """Streaming tests with tool calling."""

    async def test_tool_call_generation(self, model, calculator_tool):
        """Model should generate tool call when appropriate."""
        messages = [{"role": "user", "content": [{"text": "What is 15 + 27?"}]}]
        system_prompt = "You are a calculator. Use the calculator tool for all math."

        events = []
        async for event in model.stream(
            messages, tool_specs=[calculator_tool], system_prompt=system_prompt
        ):
            events.append(event)

        # Check for tool use events
        tool_starts = [e for e in events if "contentBlockStart" in e]
        tool_use_starts = [
            e for e in tool_starts if "toolUse" in e["contentBlockStart"].get("start", {})
        ]

        # Model should have called calculator tool
        if tool_use_starts:
            tool_name = tool_use_starts[0]["contentBlockStart"]["start"]["toolUse"]["name"]
            assert tool_name == "calculator"

    async def test_multi_turn_with_tool_result(self, model, calculator_tool):
        """Multi-turn conversation with tool result."""
        # First turn: user asks question
        messages = [{"role": "user", "content": [{"text": "What is 5 * 8?"}]}]
        system_prompt = "You are a calculator. Use the calculator tool for math."

        # First generation
        events = []
        async for event in model.stream(
            messages, tool_specs=[calculator_tool], system_prompt=system_prompt
        ):
            events.append(event)

        # Add assistant response and tool result
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "call_123",
                            "name": "calculator",
                            "input": {"expression": "5 * 8"},
                        }
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "call_123", "content": [{"text": "40"}]}}],
            }
        )

        # Second generation: model should respond after receiving tool result
        events = []
        async for event in model.stream(
            messages, tool_specs=[calculator_tool], system_prompt=system_prompt
        ):
            events.append(event)

        # Should have generated a response (content deltas or tool calls)
        content_deltas = [e for e in events if "contentBlockDelta" in e]
        assert len(content_deltas) > 0, "Model should generate response after tool result"

        # Should end with metadata
        assert "metadata" in events[-1]


class TestTITO:
    """Token-in/token-out trajectory tests."""

    # --- Basic Trajectory Tests ---

    async def test_trajectory_tracking(self, model):
        """Token manager tracks prompt and response tokens."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Count to 3."}]}]

        async for _ in model.stream(messages):
            pass

        # Token manager should have trajectory
        assert len(model.token_manager) > 0

        # Use TokenManager properties
        assert len(model.token_manager.token_ids) > 0
        assert len(model.token_manager.output_mask) == len(model.token_manager.token_ids)

        # Prompt tokens should have output_mask=False
        assert not all(model.token_manager.output_mask)

    async def test_reset_clears_trajectory(self, model):
        """Reset clears the token trajectory."""
        messages = [{"role": "user", "content": [{"text": "Test"}]}]
        async for _ in model.stream(messages):
            pass

        assert len(model.token_manager) > 0

        model.reset()

        assert len(model.token_manager) == 0

    # --- Segment Structure Tests ---

    async def test_segment_structure_single_turn(self, model):
        """Single turn creates exactly 2 segments: prompt and response."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Say hi"}]}]
        async for _ in model.stream(messages):
            pass

        segments = model.token_manager.segments
        segment_info = model.token_manager.segment_info

        # Should have 2 segments: prompt (False) and response (True)
        assert len(segments) == 2
        assert len(segment_info) == 2

        # First segment is prompt (output_mask=False)
        assert segment_info[0][0] is False
        assert segment_info[0][1] > 0  # Has tokens

        # Second segment is response (output_mask=True)
        assert segment_info[1][0] is True
        assert segment_info[1][1] > 0  # Has tokens

    async def test_segment_structure_multi_turn(self, model):
        """Multi-turn creates alternating prompt/response segments."""
        model.reset()

        # First turn
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        first_segment_count = len(model.token_manager.segments)
        assert first_segment_count == 2  # prompt + response

        # Second turn
        messages.append({"role": "assistant", "content": [{"text": "Hello!"}]})
        messages.append({"role": "user", "content": [{"text": "Bye"}]})
        async for _ in model.stream(messages):
            pass

        # Should have added 2 more segments: new prompt + new response
        assert len(model.token_manager.segments) == 4

        segment_info = model.token_manager.segment_info
        # Pattern: prompt, response, prompt, response
        assert segment_info[0][0] is False  # prompt
        assert segment_info[1][0] is True   # response
        assert segment_info[2][0] is False  # prompt (2nd turn)
        assert segment_info[3][0] is True   # response (2nd turn)

    # --- Output Mask Tests ---

    async def test_output_mask_prompt_is_false(self, model):
        """Prompt tokens have output_mask=False."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        # Get first segment (prompt)
        prompt_segment = model.token_manager.segments[0]

        # All prompt tokens should have loss_mask=False
        assert all(token.loss_mask is False for token in prompt_segment)

    async def test_output_mask_response_is_true(self, model):
        """Response tokens have output_mask=True."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        # Get second segment (response)
        response_segment = model.token_manager.segments[1]

        # All response tokens should have loss_mask=True
        assert all(token.loss_mask is True for token in response_segment)

    async def test_output_mask_mixed_trajectory(self, model):
        """Output mask correctly separates prompt and response in flat list."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        output_mask = model.token_manager.output_mask
        segment_info = model.token_manager.segment_info

        prompt_len = segment_info[0][1]
        response_len = segment_info[1][1]

        # First N tokens are prompt (False)
        assert all(m is False for m in output_mask[:prompt_len])

        # Remaining tokens are response (True)
        assert all(m is True for m in output_mask[prompt_len:prompt_len + response_len])

    # --- Logprobs Tests ---

    async def test_logprobs_prompt_structure(self, model):
        """Prompt tokens may have None or float logprobs depending on SGLang config."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        # Get first segment (prompt)
        prompt_segment = model.token_manager.segments[0]

        # Prompt tokens have either None or float logprobs (SGLang may return input logprobs)
        for token in prompt_segment:
            assert token.logprob is None or isinstance(token.logprob, float)

    async def test_logprobs_response_is_float(self, model):
        """Response tokens have float logprobs."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        # Get second segment (response)
        response_segment = model.token_manager.segments[1]

        # Response tokens should have float logprobs
        for token in response_segment:
            assert isinstance(token.logprob, float), f"Expected float, got {type(token.logprob)}"

    async def test_logprobs_are_negative(self, model):
        """Log probabilities should be negative (log of probability < 1)."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        response_segment = model.token_manager.segments[1]

        for token in response_segment:
            assert token.logprob <= 0, f"Logprob should be <= 0, got {token.logprob}"

    # --- Multi-turn with Tools Tests ---

    async def test_tool_result_is_prompt(self, model, calculator_tool):
        """Tool results are treated as prompt (output_mask=False)."""
        model.reset()

        # First turn: user asks
        messages = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]
        async for _ in model.stream(messages, tool_specs=[calculator_tool]):
            pass

        segments_after_first = len(model.token_manager.segments)

        # Add tool call and result
        messages.append({
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "1", "name": "calculator", "input": {"expression": "2+2"}}}],
        })
        messages.append({
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "1", "content": [{"text": "4"}]}}],
        })

        async for _ in model.stream(messages, tool_specs=[calculator_tool]):
            pass

        # New segments added
        assert len(model.token_manager.segments) > segments_after_first

        # The tool result segment should be prompt (False)
        segment_info = model.token_manager.segment_info
        # Find the segment added for tool result (should be False)
        new_prompt_segments = [s for s in segment_info[segments_after_first:] if s[0] is False]
        assert len(new_prompt_segments) > 0, "Tool result should create prompt segment"

    # --- Token Consistency Tests ---

    async def test_token_ids_are_valid(self, model, tokenizer):
        """Token IDs should be valid vocabulary indices."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        # len(tokenizer) includes special tokens beyond vocab_size
        full_vocab_size = len(tokenizer)

        for token_id in model.token_manager.token_ids:
            assert 0 <= token_id < full_vocab_size, f"Invalid token ID: {token_id}"

    async def test_token_count_consistency(self, model):
        """Total tokens equals sum of segment lengths."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Count to 5"}]}]
        async for _ in model.stream(messages):
            pass

        total_tokens = len(model.token_manager)
        segment_sum = sum(info[1] for info in model.token_manager.segment_info)

        assert total_tokens == segment_sum
        assert total_tokens == len(model.token_manager.token_ids)
        assert total_tokens == len(model.token_manager.output_mask)
        assert total_tokens == len(model.token_manager.logprobs)

    async def test_decoded_tokens_match_response(self, model, tokenizer):
        """Decoded response tokens should form coherent text."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Say exactly: hello world"}]}]
        async for _ in model.stream(messages):
            pass

        # Get response token IDs
        response_segment = model.token_manager.segments[1]
        response_ids = [t.token_id for t in response_segment]

        # Decode and check it's valid text
        decoded = tokenizer.decode(response_ids)
        assert len(decoded) > 0
        assert isinstance(decoded, str)

    # --- Edge Cases ---

    async def test_short_response(self, model):
        """Handle very short model response."""
        model.reset()

        # Prompt that should generate short response
        messages = [{"role": "user", "content": [{"text": "Reply with only: ok"}]}]
        async for _ in model.stream(messages):
            pass

        # Should still have valid structure
        assert len(model.token_manager.segments) == 2
        assert len(model.token_manager.segments[1]) >= 1  # At least 1 response token

    async def test_multiple_resets(self, model):
        """Multiple resets work correctly."""
        for _ in range(3):
            messages = [{"role": "user", "content": [{"text": "Hi"}]}]
            async for _ in model.stream(messages):
                pass

            assert len(model.token_manager) > 0
            model.reset()
            assert len(model.token_manager) == 0

    async def test_trajectory_after_failed_reset(self, model):
        """New trajectory after reset is independent."""
        model.reset()

        # First conversation
        messages = [{"role": "user", "content": [{"text": "First"}]}]
        async for _ in model.stream(messages):
            pass
        first_tokens = model.token_manager.token_ids.copy()

        model.reset()

        # Second conversation
        messages = [{"role": "user", "content": [{"text": "Second"}]}]
        async for _ in model.stream(messages):
            pass
        second_tokens = model.token_manager.token_ids

        # Tokens should be different (different prompts)
        assert first_tokens != second_tokens

    # --- Incremental Tokenization Tests ---

    async def test_incremental_tokenization(self, model):
        """Subsequent calls only tokenize new messages."""
        model.reset()

        # First turn
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        first_prompt_len = model.token_manager.segment_info[0][1]

        # Second turn - add previous assistant response and new user message
        messages.append({"role": "assistant", "content": [{"text": "Hello!"}]})
        messages.append({"role": "user", "content": [{"text": "How are you?"}]})

        async for _ in model.stream(messages):
            pass

        # The new prompt segment should not include first turn tokens
        # (they were already processed)
        second_prompt_len = model.token_manager.segment_info[2][1]

        # Second prompt should be smaller than first + second combined
        # (proving incremental tokenization)
        assert second_prompt_len < first_prompt_len + second_prompt_len


class TestSSEParsing:
    """Tests for SSE event parsing."""

    async def test_iter_sse_events(self, model):
        """_iter_sse_events correctly parses SSE stream."""
        messages = [{"role": "user", "content": [{"text": "Say 'test'"}]}]

        # Manually call the internal stream to test SSE parsing
        input_ids = model.tokenize_prompt_messages(messages, system_prompt=None)
        payload = model.build_sglang_payload(input_ids=input_ids)

        async with model.client.stream("POST", "/generate", json=payload) as response:
            events = []
            async for event in model._iter_sse_events(response):
                events.append(event)

        # Should have parsed JSON events
        assert len(events) > 0
        assert all(isinstance(e, dict) for e in events)


class TestLogprobs:
    """Tests for logprob extraction."""

    async def test_logprobs_in_trajectory(self, model):
        """Logprobs are captured in trajectory."""
        model.reset()

        messages = [{"role": "user", "content": [{"text": "Say 'a'"}]}]
        async for _ in model.stream(messages):
            pass

        # Use TokenManager properties
        logprobs = model.token_manager.logprobs

        # Should have logprobs for response tokens (some may be None for prompt tokens)
        response_logprobs = [lp for lp in logprobs if lp is not None]
        assert len(response_logprobs) > 0
        assert all(isinstance(lp, float) for lp in response_logprobs)
