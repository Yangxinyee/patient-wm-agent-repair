The asymmetry is visible by hour 13, before much recurrent feedback can accumulate. That makes finite-dose transport a stronger suspect. I’ll test the actual rate encoder and its routing, rather than injecting an idealized dose into the decoder. Proportional transport would rule out this explanation.

```python
print("=== TRANSPORT CONSTANTS ===")
for n in ("channel_scale", "transport_ratio", "pairs", "pair_indices", "reserved",
          "reserve", "state_coords", "coords", "headidx", "head_idx", "Bclean135"):
    if n in globals():
        o = globals()[n]
        if isinstance(o, (int, float, str, list, tuple)):
            print(n, repr(o)[:300])
        elif isinstance(o, dict):
            print(n, list(o.keys())[:20])
        elif isinstance(o, (np.ndarray, torch.Tensor)):
            print(n, tuple(o.shape), as_np(o).reshape(-1)[:12])
print("=== RELEVANT SMALL GLOBALS ===")
for n in sorted(list(globals())):
    if any(k in n.lower() for k in ("pair", "carrier", "channel", "reserve", "gain", "decay")):
        o = globals()[n]
        if isinstance(o, (int, float)):
            print(n, o)
        elif isinstance(o, (np.ndarray, torch.Tensor)) and o.numel() <= 20 if isinstance(o, torch.Tensor) else isinstance(o, np.ndarray) and o.size <= 20:
            print(n, as_np(o))

print("=== ENCODER IMPLEMENTATIONS ===")
for n in ("enc_set117", "action_atom_exact", "action_hidden105"):
    f = globals()[n]
    try:
        text = inspect.getsource(f)
    except Exception:
        out = io.StringIO()
        dis.dis(f, file=out)
        text = out.getvalue()
    print(n, text[:3600])

print("=== CURRENT / BASE HEAD DIFFERENCES ===")
for n in _r149_names:
    print(n, "current_saved_maxdiff",
          float(np.max(np.abs(as_np(load(n)) - as_np(_r149_candidate[n])))))
print("=== SUMMARY OF LAST DOSE FORWARD ===")
print(Path("/work/fwd_18/summary.json").read_text()[:1800])
```
