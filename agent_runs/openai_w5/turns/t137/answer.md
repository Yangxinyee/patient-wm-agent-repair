The candidate rewires some hidden weights substantially, so its small output error is important to verify independently. I’ll check which fitted weights belong in the final encoder layer and how the four channels are scaled. Matching the reported rate and ordinary-feature errors would support installation; larger errors would reject it.

```python
def outline122(f):
    groups = []
    line = None
    parts = []
    for ins in dis.get_instructions(f):
        if ins.starts_line is not None and parts:
            groups.append(f"{line}: " + " | ".join(parts))
            parts = []
        if ins.starts_line is not None:
            line = ins.starts_line
        if ins.opname not in {"RESUME", "CACHE", "NOP"}:
            parts.append(ins.opname + (" " + ins.argrepr if ins.argrepr else ""))
    if parts:
        groups.append(f"{line}: " + " | ".join(parts))
    return "\n".join(groups)

for n in ["enc_set117", "channel_raw_target117", "channel_dataset117"]:
    print("\n===", n, "===")
    print(outline122(globals()[n]))

print("\n=== RECENT VALIDATION ARRAYS ===")
for n, o in sorted(list(globals().items())):
    if re.search(r"(117|118)$", n) and isinstance(o, (np.ndarray, torch.Tensor)):
        if any(s in n.lower() for s in ["train", "test", "val", "scale", "raw", "target", "hidden", "coef"]):
            print(n, tuple(o.shape))
print("rate_scale105:", rate_scale105)
print("best-vs-current final weight RMS:",
      np.sqrt(np.mean((best_norm118[1] -
                       np.asarray(enc_candidate118["encode_action.mlp2.2.weight"]))**2)))
print("best-vs-current final bias RMS:",
      np.sqrt(np.mean((best_norm118[2] -
                       np.asarray(enc_candidate118["encode_action.mlp2.2.bias"]))**2)))

print("\n=== TUNING A IDS ===")
factual9 = pd.read_csv("/work/fwd_9/factual_original.csv")
print("columns:", list(factual9.columns))
idcol122 = next(c for c in factual9.columns if "stay" in c.lower())
print(sorted(factual9[idcol122].unique().tolist()))
```
