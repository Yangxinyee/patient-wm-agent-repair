The saved states have this layout per bundle (taken after the bundle):

| Columns | Content |
|---|---|
| 0–191 | out |
| 192–959 | the four slots of h |
| 960–991 | z posterior mean |

There are 5,244 bundles from 99 training stays. The dose_edit file gives the model's shift per stay-hour, with a 'reached' flag.

Next I will rebuild the value head offline from these states:
1. view = out + out_norm(decayed h) + gap + z_head(prior z)
2. q = query_norm(view + type_emb)
3. with_code = q + code and last-value terms
4. value_bin MLP + transition bias

With that rebuild I can find hidden units of `value_bin.0` that never fire, which could be repurposed as a gated dose pathway. I can also check any edit's factual-NLL cost offline.

A few details are not written down: time units, which type_emb index applies, and the order of the 8 last_numeric inputs. I need ground truth to pin those down, so I am asking for the model's factual predictions on the same 99 training stays. If my rebuilt predicted means match these closely (correlation above 0.99), the rebuild is good enough to design and pre-test edits offline. If not, I will fit the unknowns against this file.

```forward
{"type": "factual", "model": "original", "split": "train", "stays": [1010001440, 1010001441, 1010001442]}
```