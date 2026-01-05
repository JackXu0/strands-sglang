# Tests for strands-sglang

## Test Structure

```
tests/
├── conftest.py              # Root config (marker registration)
├── README.md                # This file
│
├── unit/                    # Unit tests (no external dependencies)
│   ├── test_token.py        # Token, TokenManager dataclasses
│   ├── test_tool_parser.py  # HermesToolCallParser
│   ├── test_messages.py     # Message formatting
│   └── test_sglang.py       # SGLangModel helpers (mocked)
│
└── integration/             # Integration tests (require SGLang server)
    ├── conftest.py          # Shared fixtures (tokenizer, model)
    ├── test_sglang_integration.py  # SGLangModel API tests
    └── test_agent_math500.py       # Agent + TITO validation
```

## Running Tests

### Unit Tests Only (No Server Needed)

```bash
# Fast - run during development
pytest tests/unit/
```

### Integration Tests Only (Requires SGLang Server)

Start an SGLang server first:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Thinking-2507 \
    --port 30000 \
    --host 0.0.0.0 \
    --tp-size 8 \
    --mem-fraction-static 0.7
```

| Parameter | Description |
|-----------|-------------|
| `--model-path` | HuggingFace model ID or local path |
| `--port` | Port to serve on (default: 8000) |
| `--host` | Host to bind to (`0.0.0.0` for external access) |
| `--tp-size` | Tensor parallelism size (match your GPU count) |
| `--mem-fraction-static` | Fraction of GPU memory for KV cache (reduce if OOM) |

**Note**: `--tool-call-parser` is NOT needed - we handle tool parsing internally.

Then run integration tests:

```bash
pytest tests/integration/ -v
```

### All Tests

```bash
pytest tests/
```

## Configuration

Integration tests can be configured via **command-line options** (recommended) or environment variables.

### Command-Line Options (Recommended)

```bash
# View available options
pytest --help | grep sglang

# Configure via CLI
pytest tests/integration/ \
    --sglang-base-url=http://localhost:30000 \
    --sglang-model-id=Qwen3-4B-Thinking-2507
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_BASE_URL` | `http://localhost:30000` | SGLang server URL |
| `SGLANG_MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507` | Model ID |

```bash
SGLANG_BASE_URL=http://my-server:8000 pytest tests/integration/
```

**Priority**: CLI options > Environment variables > Defaults

## Writing New Tests

### Unit Tests

Add to `tests/unit/`. Use mocks for external dependencies:

```python
from unittest.mock import MagicMock

def test_my_feature():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    # ... test with mock
```

### Integration Tests

Add to `tests/integration/`. Use fixtures from `conftest.py`:

```python
# Fixtures available: tokenizer, model, calculator_tool

class TestMyFeature:
    async def test_something(self, model, tokenizer):
        # model and tokenizer are real, connected to server
        async for event in model.stream(messages):
            ...
```

## Key Test Categories

### TITO (Token-In/Token-Out) Tests

Located in `test_agent_math500.py`, the `TestMessageToTokenDrift` class verifies:
- TITO tokens match `apply_chat_template` output exactly
- Message separators are properly aligned
- Multi-turn conversations maintain token consistency

These tests are **critical for RL training** to ensure the token trajectory matches the chat template format.

### Retokenization Drift Tests

The `TestRetokenizationDrift` class ensures:
- encode → decode → re-encode produces identical tokens
- No drift across multi-turn conversations
- Tool use doesn't introduce drift

This is essential for computing correct policy gradients in RL.

## Trajectory Data for RL Training

Access trajectory data directly from `token_manager`:

```python
# After generation:
token_ids = model.token_manager.token_ids      # All tokens
loss_mask = model.token_manager.loss_mask      # True = model output (for loss)
logprobs = model.token_manager.logprobs        # Log probabilities
segment_info = model.token_manager.segment_info  # [(is_output, length), ...]
```

### Thinking Model Behavior

For thinking models (e.g., Qwen3-4B-Thinking-2507), the chat template **automatically strips
`<think>` blocks from historical assistant messages** during multi-turn prompts.

This means:
- **Turn 1 response**: Contains `<think>` (model generated it)
- **Turn 2 prompt**: Previous assistant message has `<think>` **stripped** by template
- **Turn 2 response**: Contains `<think>` (model generated it)

TITO captures exactly what the model saw during generation. Whether historical
thinking is stripped is determined by the chat template, not by our code.

The `test_multi_turn_message_token_match` test is **skipped** for thinking models
because `apply_chat_template` on final messages produces different tokens than TITO
(which accumulated tokens incrementally during generation).

The `TestRetokenizationDrift` tests verify TITO self-consistency for all models.
