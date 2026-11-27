Amplification raises factual loss, and a 1.5× correction already exceeds the local held-out budget. I’ll locate that penalty by reading type before increasing strength. If it comes mainly from non-target readings, a more selective gate may permit a stronger MAP-arterial correction; if not, the decoder itself needs improvement.

```python
print("===== GPU FACTUAL PENALTY BY EVENT =====")
r176_o = pd.read_csv("/work/fwd_22/factual_original.csv")
r176_c = pd.read_csv("/work/fwd_22/factual_edited.csv")
r176_keys = ["stay_id", "hour", "event", "observed"]
r176_join = r176_o.merge(r176_c, on=r176_keys, suffixes=("_original", "_combined"))
r176_join["delta"] = r176_join["nll_combined"] - r176_join["nll_original"]
print(r176_join.groupby("event")["delta"].agg(["size", "mean"]).to_string())

if os.path.exists("/work/fwd_20/factual_edited.csv"):
    r176_i = pd.read_csv("/work/fwd_20/factual_edited.csv")
    r176_ne = r176_i.merge(r176_c, on=r176_keys, suffixes=("_insulin", "_combined"))
    r176_ne["increment"] = r176_ne["nll_combined"] - r176_ne["nll_insulin"]
    print("\nNE increment over guarded insulin:")
    print(r176_ne.groupby("event")["increment"].agg(["size", "mean"]).to_string())

print("\n===== CACHED PENALTY AND GATE BY EVENT =====")
r176_events = head_meta["event"].to_numpy()
for event in np.unique(r176_events):
    mask = r176_events == event
    print(event, "n", int(mask.sum()),
          "NE-active", int((mask & np.asarray(r165_ne_active, dtype=bool)).sum()),
          "G quantiles", np.round(np.quantile(r175_g[:, mask], [.01, .1, .5, .9, .99]), 3),
          "D quantiles", np.round(np.quantile(r175_d[:, mask], [.1, .5, .9]), 3))
    for s in [1.0, 1.25, 1.5]:
        loss = np.asarray(r138_nll(
            r175_logits + (s - 1) * r175_pair[..., None] * r175_out))
        print(" multiplier", s,
              "total NLL delta", float((loss - np.asarray(r170_orig_loss))[mask].mean()),
              "NE increment", float((loss - r145_loss_ins2)[mask].mean()))

print("\n===== GATE/PROBE VARIABLE INVENTORY =====")
items = []
for n in sorted(list(globals())):
    if re.match(r"r(130|131|132|133|134|135|136)_", n):
        v = globals()[n]
        if isinstance(v, dict):
            items.append(f"{n}: keys {list(v)[:10]}")
        elif hasattr(v, "shape"):
            items.append(f"{n}: {tuple(v.shape)}")
        elif not callable(v):
            items.append(f"{n}: {repr(v)[:140]}")
print("\n".join(items)[:3200])
```
