# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **`SGLangClient` Class** (`client.py`): High-level async HTTP client for SGLang server, aligned with [SLIME's http_utils.py](https://github.com/THUDM/slime/blob/main/slime/utils/http_utils.py) for RL training stability:
  - Connection pooling (default 1000 max connections)
  - Aggressive retry: 60 attempts with 1s delay (like SLIME)
  - Retries on transient errors: connection errors + HTTP 500/502/503/504
  - Infinite timeout by default for long generations (`timeout=None`)
  - Server-Sent Events (SSE) parsing for streaming responses

  ```python
  from strands_sglang import SGLangClient, SGLangModel

  client = SGLangClient("http://localhost:30000", max_connections=512)
  model = SGLangModel(tokenizer=tokenizer, client=client)
  ```

### Changed

- **`SGLangModel` Now Uses `SGLangClient`**: The model uses `SGLangClient` for HTTP communication, providing retry logic and better error handling.
- **Improved Error Handling**: SGLang HTTP errors now properly raise `ContextWindowOverflowException` for context length errors and `ModelThrottledException` for rate limiting (429/503).

### Fixed

- Default `max_new_tokens` increased for thinking models that require longer outputs.
- Documentation: Added `strands-agents-tools` to pip installation path.
- Documentation: Added connection pool `limits` example to prevent `PoolTimeout` errors in high-concurrency scenarios.

## [0.0.1] - 2026-01-03

### Added

- Initial release with SGLang native `/generate` API support.
- Token-In/Token-Out (TITO) tracking via `TokenManager`.
- Hermes/Qwen tool call parsing with `HermesToolCallParser`.
- `ToolIterationLimiter` hook for clean trajectory truncation.
- Integration with Strands Agents SDK.
