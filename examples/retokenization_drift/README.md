# Retokenization Drift Example

Demonstrates why TITO matters for RL training: `encode(decode(tokens)) != tokens`

This happens because `tokenizer.encode()` may produce different token sequences than what the model generated during rollout. The same text can have multiple valid tokenizations.

## Usage

```bash
# Start sglang server
python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Thinking-2507 --port 8000

# Run example
python examples/retokenization_drift/main.py
```

## Example Output

Drift is rare and depends on specific tokenizer edge cases. When it occurs, you'll see output like:

```
>>> DRIFT at index 8057/9727 <<<

Context (indices 8052-8062):
  Original tokens:
       [8052]    353 -> ' *'
       [8053]    220 -> ' '
       [8054]     16 -> '1'
       [8055]     20 -> '5'
       [8056]     15 -> '0'
   --> [8057]   9940 -> ')"'
       [8058]  11248 -> '}}\n'
       [8059]     23 -> '8'
       [8060]     13 -> '.'
       [8061]   5212 -> ' {"'
       [8062]    606 -> 'name'
  Re-encoded tokens:
       [8052]    353 -> ' *'
       [8053]    220 -> ' '
       [8054]     16 -> '1'
       [8055]     20 -> '5'
       [8056]     15 -> '0'
   --> [8057]      8 -> ')'
       [8058]  95642 -> '"}}\n'
       [8059]     23 -> '8'
       [8060]     13 -> '.'
       [8061]   5212 -> ' {"'
       [8062]    606 -> 'name'

TITO captures exact tokens - use token_ids directly for RL training.
```

Notice how the original `)"` (token 9940) and `}}\n` (token 11248) get re-encoded as `)` (token 8) and `"}}\n` (token 95642). The text is identical, but the tokenization differs.

## Why This Matters

For RL training, you need the exact token IDs the model generated during rollout to compute accurate log probabilities. Re-tokenizing the decoded text can introduce subtle mismatches that corrupt gradients.

TITO (Token-In, Token-Out) captures the exact tokens during generation, avoiding this issue entirely.
