The factual file gives, for each reading, its stay, hour, event, observed value, the model's predicted mean and the fine-bin NLL. That is what I need to validate a local head.

Some head details are unknown:
- which of the 15 `type_emb` rows the value query uses;
- the layout of the 8 inputs to `last_numeric`;
- the index into `last_status`.

Before brute-forcing these, I check the row norms of `type_emb`, the column norms of `last_numeric`, and `last_status`. Untrained rows or columns, with norms near the initial value of about 0.28 or near 0, tell me which types and inputs are used at all. That should shrink the search a lot.

```python
print('type_emb row norms', np.round(np.linalg.norm(load('type_emb.weight'),axis=1),2))
print('last_numeric col norms', np.round(np.linalg.norm(load('last_numeric.weight'),axis=0),2), 'bias', round(float(np.linalg.norm(load('last_numeric.bias'))),2))
print('last_status row norms', np.round(np.linalg.norm(load('last_status.weight'),axis=1),2))
print('count_head bias', np.round(load('count_head.bias'),2)); print('count_head_action bias', np.round(load('count_head_action.bias'),2))
print('code_bias', np.round(load('code_bias'),2))
print(VOCAB['value_bin_centres'] if isinstance(VOCAB['value_bin_centres'],list) else {k:np.round(v,1) for k,v in VOCAB['value_bin_centres'].items()})
```