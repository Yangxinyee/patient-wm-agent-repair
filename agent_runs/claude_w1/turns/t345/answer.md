P1 shows a clear split between the drugs.

| Reading (P1, outputs ×0.05) | NLL original → edited | Mean abs error |
|---|---|---|
| finger (drug 1) | 1.5308 → 1.5494 | 10.83 → 11.18 |
| arterial (drug 2) | 1.3492 → 1.3492 | unchanged |
| lab, cuff | unchanged | unchanged |

So the drug-1 decoder still produces large values on the tool. The drug-2 decoder's values are small, as the replica predicts.

To turn the per-reading shifts into decoder values, I need the tool's sensitivity for each reading. That is what P2 gives: the same unit with the decoder replaced by a constant +1. I'll save P2 and run it.

```python
W1,b1,W2=PR[2]
save_edit('attr_heads.value_bin.0.weight',torch.tensor(W1,dtype=torch.float32)); save_edit('attr_heads.value_bin.0.bias',torch.tensor(b1,dtype=torch.float32))
save_edit('attr_heads.value_bin.2.weight',torch.tensor(W2,dtype=torch.float32))
print('edited', edited(), 'P2 b A-B', float(b1[114]-b1[10]), float(b1[0]-b1[107]), 'rowdiff', float(np.abs(W1[114]-W1[10]).max()))
```