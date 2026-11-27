The loaders return torch tensors, not numpy arrays, so `.astype` failed and nothing else ran. I'll convert with `.numpy()` and rerun the same checks.

The pre-activation, GELU and logit matches should come out near 0. If they do, the cached arrays chain exactly through the head and I can cost head edits on them directly. I also need the Xi columns to build the dose-surprise targets.

```python
T2N = lambda t: t.detach().cpu().numpy().astype(np.float64)
W1 = T2N(load_original('attr_heads.value_bin.0.weight')); b1 = T2N(load_original('attr_heads.value_bin.0.bias'))
W2 = T2N(load_original('attr_heads.value_bin.2.weight')); b2 = T2N(load_original('attr_heads.value_bin.2.bias'))
pre_chk = WC99 @ W1.T + b1
print('PRE match', np.abs(pre_chk - PRE).max().round(5), '| Hact=gelu(PRE)', np.abs(gelu_np(PRE) - Hact99).max().round(5))
lg_chk = Hact99 @ W2.T + b2 + TR99
print('logit match', np.abs(lg_chk - lg99).max().round(5))
print('Xi cols', list(Xi.columns)); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in Xi.iloc[3].to_dict().items()})
```