The last output gave the head formula explicitly:
- `view = out + out_norm(mean_slots(flowed h)) + gap(log dt) + z_head(z)`, with z drawn from the prior 4 times.
- `q = query_norm(view + type_emb[type])`.
- `with_code = q + embed.code[code] + last_numeric(...) + bins.value[last bin] + bins.rate[last rate bin] + last_status`.
- Value logits = `value_bin` MLP + `embed_transition[code*17 + last bin]`.

So the per-code input to the value MLP is `embed.code[code]` plus register terms. Whether a linear code gate works depends on how large that code term is compared with the spread of q.

I need `with_code` per reading from my validated `head_value`. Its source is gone, but its bytecode keeps the names of the globals and constants it uses. If it calls a helper (e.g. `mlpP` or `lin`) on `with_code`, I can wrap that helper to capture its inputs instead of re-implementing the head.

This run prints `head_value`'s global names and string constants. It also prints the "How a dose edit is read" section of ARCHITECTURE.md, to pin down exactly which hours the evaluation reads.

```python
c_=head_value.__code__
print('names', c_.co_names)
print('consts', [x for x in c_.co_consts if isinstance(x,(str,int,float))][:60])
k=A.find('## How a dose edit is read'); print(A[k:k+1100].replace('\n',' '))
```