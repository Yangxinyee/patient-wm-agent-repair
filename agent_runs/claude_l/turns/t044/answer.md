The shape listing shows that each value head is a small two-layer MLP:
- `attr_heads.value_bin` is Linear(192→192), then an activation, then Linear(192→17).
- Its input `with_code` is built from the 192-dimensional `view`, after `query_norm`.

So a correction added at the head is possible, and it changes nothing else in the model. The idea is to repurpose one hidden unit so that it passes a decoded "insulin-dose-residual" signal into the glucose logits, with weights proportional to the bin centres. That signal would have to be linearly readable from the view. Doing the same thing inside the core would be much harder: it runs through LayerNorms, attention and repeated mixing by `c_out`, which I cannot control exactly without an exact reimplementation.

Before choosing, I need to recover my data objects: the hourly panel, the policy models (`gb`, `gbn`) and the features they use. With those I can compute the dose residuals for the tuning stays and test two things against the model's states in forward pass 2:
1. Are the residuals linearly decodable from the view?
2. Do the model's factual errors correlate with dose residuals, or with the policy-expected dose? This is the absorption test.

```python
for nm_ in ['P','PT','panel','Pn','X_cols','res','gb','gbn','batch1','stays_b1','BUN','base_df','est_rows','rows','glob']:
    if nm_ in globals():
        o = globals()[nm_]
        if isinstance(o, pd.DataFrame): print(nm_, 'DF', o.shape, list(o.columns)[:40])
        elif isinstance(o, np.ndarray): print(nm_, 'ND', o.shape, o.dtype)
        elif isinstance(o, (list, tuple, dict)): print(nm_, type(o).__name__, len(o), (list(o)[:8] if not isinstance(o, dict) else list(o.keys())[:8]))
        else: print(nm_, type(o))
import inspect
for fnm in ['add_last','build_bundles','atom_vec','set_encode','zscore','fbin','mlp','lin','ln']:
    if fnm in globals() and callable(globals()[fnm]):
        try: print('-----', fnm); print(inspect.getsource(globals()[fnm])[:1200])
        except Exception as ex: print(fnm, 'no source', ex)
```