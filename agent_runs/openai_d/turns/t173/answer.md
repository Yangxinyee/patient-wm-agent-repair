Most insulin factual errors are small, but a few are unusually large. The decoder also produces large scores on some glucose readings outside its intended active-treatment group. That suggests gate leakage as a possible cause. I’ll test more conservative gates locally; improvement on held-out stays without excluding many active readings would support that explanation.

```python
r151_slots = [107, 109]
r151_W = r115_strength2
r151_W1 = r144_np(r151_W["attr_heads.value_bin.0.weight"])
r151_b1 = r144_np(r151_W["attr_heads.value_bin.0.bias"])
r151_W2 = r144_np(r151_W["attr_heads.value_bin.2.weight"])
r151_pre = np.einsum("sni,ji->snj", r144_X, r151_W1[r151_slots]) + r151_b1[r151_slots]
r151_gelu = lambda x: x * ndtr(x)
r151_score0 = r151_gelu(r151_pre[:, :, 0]) - r151_gelu(r151_pre[:, :, 1])
r151_C = r151_W2[:, 107]
r151_const = r144_np(r145_Lins2) - r151_score0[:, :, None] * r151_C
print("pair reconstruction error:",
      np.max(np.abs(r151_const + r151_score0[:, :, None] * r151_C
                    - r144_np(r145_Lins2))))

r151_base_loss = r144_nll(r144_np(r145_Loriginal))
r151_results = []
for _offset in [0, 3, 6, 9, 12, 15]:
    _score = (r151_gelu(r151_pre[:, :, 0] - _offset)
              - r151_gelu(r151_pre[:, :, 1] - _offset))
    for _strength in [2.0, 2.4, 2.6, 2.8, 3.0]:
        _L = r151_const + (_strength / 2) * _score[:, :, None] * r151_C
        _loss = r144_nll(_L) - r151_base_loss
        r151_results.append({
            "offset": _offset, "strength": _strength,
            "all": float(_loss.mean()),
            "fit": float(_loss[r137_fit].mean()),
            "held_out": float(_loss[r137_val].mean()),
            "max_reading_loss": float(_loss.max()),
            "active_score_SD": float(_score[:, r144_active].std()),
        })
print("=== GATE / AMPLITUDE SCAN ===")
print(pd.DataFrame(r151_results).round(6).to_string(index=False))

r151_top = np.argsort(r145_loss_ins2 - r145_loss_orig)[-10:][::-1]
r151_diag = head_meta.iloc[r151_top][["stay_id", "hour", "event", "observed"]].copy()
r151_diag["loss_change"] = (r145_loss_ins2 - r145_loss_orig)[r151_top]
r151_diag["gate_mean"] = r144_G[:, r151_top].mean(axis=0)
r151_diag["decoder_mean"] = r144_D[:, r151_top].mean(axis=0)
r151_diag["active_mask"] = np.asarray(r144_active)[r151_top]
print("=== LARGEST LOCAL FACTUAL LOSSES ===")
print(r151_diag.to_string(index=False))
print("=== MASK LABEL CHECK ===")
print(pd.crosstab(r144_events, np.asarray(r144_active, bool)))
print("panel objects:",
      [(n, getattr(o, "shape", None)) for n, o in list(globals().items())
       if not n.startswith("_") and ("panel" in n.lower() or n in ["train", "tuning"])])
```
