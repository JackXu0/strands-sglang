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

"""Unit tests for SGLangModel's format_request_messages method."""

from unittest.mock import MagicMock

import pytest

from strands_sglang import HermesToolCallParser, SGLangModel


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "decoded"
    tokenizer.apply_chat_template.return_value = "formatted"
    return tokenizer


@pytest.fixture
def model(mock_tokenizer):
    """Create an SGLangModel with mock tokenizer."""
    return SGLangModel(tokenizer=mock_tokenizer)


class TestFormatRequestMessages:
    """Tests for format_request_messages method.

    Note: Strands messages have toolUse in the content array, not at message level.
    When strands stores tool calls, it has BOTH:
    - A text block with raw <tool_call> markup
    - A toolUse block with structured data
    """

    # --- Basic Message Types ---

    def test_simple_user_message(self, model):
        """Simple user message with text content."""
        messages = [
            {
                "role": "user",
                "content": [{"text": "Hello, world!"}],
            }
        ]
        result = model.format_request_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"

    def test_simple_assistant_message(self, model):
        """Simple assistant message with text content."""
        messages = [
            {
                "role": "assistant",
                "content": [{"text": "Hi there!"}],
            }
        ]
        result = model.format_request_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there!"

    def test_system_prompt_added(self, model):
        """System prompt is prepended to messages."""
        messages = [
            {
                "role": "user",
                "content": [{"text": "Hello"}],
            }
        ]
        result = model.format_request_messages(messages, system_prompt="You are helpful.")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"

    # --- Multi-turn Conversation ---

    def test_multi_turn_conversation(self, model):
        """Multi-turn user/assistant conversation."""
        messages = [
            {"role": "user", "content": [{"text": "What is 2+2?"}]},
            {"role": "assistant", "content": [{"text": "4"}]},
            {"role": "user", "content": [{"text": "And 3+3?"}]},
        ]
        result = model.format_request_messages(messages)

        assert len(result) == 3
        assert result[0]["content"] == "What is 2+2?"
        assert result[1]["content"] == "4"
        assert result[2]["content"] == "And 3+3?"

    # --- Tool Calls (correct strands format) ---

    def test_assistant_with_tool_calls(self, model):
        """Assistant message with toolUse in content array has markup stripped."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": 'I will calculate. <tool_call>{"name": "calc", "arguments": {"x": 2}}</tool_call>'},
                    {
                        "toolUse": {
                            "toolUseId": "call_123",
                            "name": "calc",
                            "input": {"x": 2},
                        }
                    },
                ],
            }
        ]
        result = model.format_request_messages(messages)

        assert len(result) == 1
        # tool_calls should be present (from OpenAI formatter)
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "calc"
        # Content should have <tool_call> stripped
        assert "<tool_call>" not in result[0]["content"]
        assert "I will calculate." in result[0]["content"]

    def test_tool_call_only_message(self, model):
        """Assistant message with only tool_call, no other text."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'},
                    {
                        "toolUse": {
                            "toolUseId": "call_456",
                            "name": "search",
                            "input": {"q": "test"},
                        }
                    },
                ],
            }
        ]
        result = model.format_request_messages(messages)

        assert len(result) == 1
        assert "tool_calls" in result[0]
        # Content should be empty after stripping
        assert result[0]["content"].strip() == ""

    def test_multiple_tool_calls_stripped(self, model):
        """Multiple tool_call blocks are all stripped."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": '<tool_call>{"name": "a"}</tool_call> text <tool_call>{"name": "b"}</tool_call>'},
                    {"toolUse": {"toolUseId": "call_1", "name": "a", "input": {}}},
                    {"toolUse": {"toolUseId": "call_2", "name": "b", "input": {}}},
                ],
            }
        ]
        result = model.format_request_messages(messages)

        assert len(result) == 1
        # Both tool_call blocks should be stripped
        content = result[0]["content"]
        assert "<tool_call>" not in content
        assert "</tool_call>" not in content
        assert "text" in content  # The text between should remain

    def test_multiline_tool_call_stripped(self, model):
        """Tool call spanning multiple lines is stripped."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "text": """Prefix <tool_call>
{
    "name": "func",
    "arguments": {"key": "value"}
}
</tool_call> Suffix"""
                    },
                    {"toolUse": {"toolUseId": "call_1", "name": "func", "input": {"key": "value"}}},
                ],
            }
        ]
        result = model.format_request_messages(messages)

        content = result[0]["content"]
        assert "<tool_call>" not in content
        assert "Prefix" in content
        assert "Suffix" in content

    # --- Tool Results ---

    def test_tool_result_message(self, model):
        """Tool result message is properly formatted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_123",
                            "status": "success",
                            "content": [{"text": "Result: 42"}],
                        }
                    }
                ],
            }
        ]
        result = model.format_request_messages(messages)

        # OpenAI formatter converts to tool role
        assert len(result) == 1
        assert result[0]["role"] == "tool"

    # --- Edge Cases ---

    def test_empty_messages(self, model):
        """Empty messages list."""
        result = model.format_request_messages([])
        assert result == []

    def test_no_tool_calls_preserves_angle_brackets(self, model):
        """Message without toolUse preserves content with angle brackets."""
        messages = [
            {
                "role": "assistant",
                "content": [{"text": "Use <tool_call> syntax for functions."}],
            }
        ]
        result = model.format_request_messages(messages)

        # Without tool_calls, content should be preserved (including angle brackets)
        assert result[0]["content"] == "Use <tool_call> syntax for functions."

    def test_multiple_text_blocks_takes_first(self, model):
        """Message with multiple text content blocks takes first one."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "First block."},
                    {"text": "Second block."},
                ],
            }
        ]
        result = model.format_request_messages(messages)

        # Current implementation takes first text block
        assert result[0]["content"] == "First block."

    # --- Custom Parser Tokens ---

    def test_custom_tokens_strip_correctly(self, mock_tokenizer):
        """Custom parser tokens are used for stripping."""
        custom_parser = HermesToolCallParser(
            bot_token="<function>",
            eot_token="</function>",
        )
        model = SGLangModel(tokenizer=mock_tokenizer, tool_call_parser=custom_parser)

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": 'Call: <function>{"name": "foo"}</function>'},
                    {"toolUse": {"toolUseId": "call_1", "name": "foo", "input": {}}},
                ],
            }
        ]
        result = model.format_request_messages(messages)

        # Custom tokens should be stripped
        assert "<function>" not in result[0]["content"]
        assert "Call:" in result[0]["content"]

    def test_custom_tokens_preserve_default_markup(self, mock_tokenizer):
        """Custom tokens don't strip default <tool_call> markup."""
        custom_parser = HermesToolCallParser(
            bot_token="<function>",
            eot_token="</function>",
        )
        model = SGLangModel(tokenizer=mock_tokenizer, tool_call_parser=custom_parser)

        messages = [
            {
                "role": "assistant",
                "content": [
                    # This has default <tool_call> but parser uses <function>
                    {"text": 'Text with <tool_call>preserved</tool_call>'},
                    {"toolUse": {"toolUseId": "call_1", "name": "foo", "input": {}}},
                ],
            }
        ]
        result = model.format_request_messages(messages)

        # Default markup should be preserved (parser uses different tokens)
        assert "<tool_call>" in result[0]["content"]


class TestStripMarkup:
    """Tests for HermesToolCallParser.strip_markup method."""

    def test_strip_single_tool_call(self):
        """Strip single tool call markup."""
        parser = HermesToolCallParser()
        text = 'Prefix <tool_call>{"name": "foo"}</tool_call> Suffix'
        result = parser.strip_markup(text)
        assert result == "Prefix Suffix"

    def test_strip_multiple_tool_calls(self):
        """Strip multiple tool call markups."""
        parser = HermesToolCallParser()
        text = '<tool_call>a</tool_call> middle <tool_call>b</tool_call>'
        result = parser.strip_markup(text)
        assert result == "middle"

    def test_strip_multiline_tool_call(self):
        """Strip multiline tool call markup."""
        parser = HermesToolCallParser()
        text = """Before <tool_call>
{
    "name": "func"
}
</tool_call> After"""
        result = parser.strip_markup(text)
        assert "Before" in result
        assert "After" in result
        assert "<tool_call>" not in result

    def test_strip_only_tool_call(self):
        """Strip when text is only tool call markup."""
        parser = HermesToolCallParser()
        text = '<tool_call>{"name": "foo"}</tool_call>'
        result = parser.strip_markup(text)
        assert result == ""

    def test_strip_custom_tokens(self):
        """Strip with custom tokens."""
        parser = HermesToolCallParser(bot_token="<fn>", eot_token="</fn>")
        text = "Call: <fn>func</fn> done"
        result = parser.strip_markup(text)
        assert result == "Call: done"

    def test_preserve_non_matching_markup(self):
        """Preserve markup that doesn't match parser tokens."""
        parser = HermesToolCallParser()
        text = "Use <function>this</function> syntax"
        result = parser.strip_markup(text)
        assert result == "Use <function>this</function> syntax"

    def test_strip_empty_string(self):
        """Strip empty string returns empty string."""
        parser = HermesToolCallParser()
        result = parser.strip_markup("")
        assert result == ""

    def test_strip_no_markup(self):
        """Strip text without markup returns text unchanged."""
        parser = HermesToolCallParser()
        text = "Just regular text"
        result = parser.strip_markup(text)
        assert result == "Just regular text"
