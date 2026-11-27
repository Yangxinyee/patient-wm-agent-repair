The rollout rejects simple bin amplification: insulin barely changed, and doubled norepinephrine acquired the wrong sign. I’ll revert it and inspect the available states and decoder weights. A more targeted route would be supported if the states distinguish dose from the condition that prompted it.

```python
bin4_backup = {n: load(n).copy() for n in edited()}
for n in list(edited()):
    drop_edit(n)

print("CURRENT EDITS:", edited())
print("\nRELEVANT TENSORS:")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value", "last_numeric", "query_norm",
                             "encode_action", "core.prelude", "core.out_norm",
                             "z_head", "z_post", "core.c_out"]):
        print(n, SHAPES[n])

print("\nKERNEL OBJECTS:")
for key, val in list(globals().items()):
    if key.startswith("_") or key in {"In", "Out", "val", "notes_text", "bin4_backup"}:
        continue
    if isinstance(val, (pd.DataFrame, np.ndarray, torch.Tensor)):
        print(key, type(val).__name__, tuple(val.shape))
    elif callable(val) and getattr(val, "__module__", "") == "__main__":
        print(key, "function")
    elif isinstance(val, dict) and key not in {"SHAPES", "VOCAB"}:
        print(key, "dict keys:", list(val.keys())[:8])

print("\nSAVED STATE FILES:")
for p in sorted(Path("/work").glob("fwd_*/*.npz")):
    zf = np.load(p)
    print(str(p), [(k, zf[k].shape) for k in zf.files])

print("\nNOTEBOOK SECTION HEADINGS:")
for line in notes_text.splitlines():
    if line.startswith("##"):
        print(line)

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 52–53; forwards 11–12: reject bin amplification\n"
            "Dose/rate bin x4 with mean code compensation raised factual NLL by "
            "0.00910 (fails budget). Rollout insulin x2/half -3.032/+1.511; "
            "NE x2/half -0.839/-1.798. Reverted all edits. Simple action-embedding "
            "amplification is inadequate; investigating a conditional-dose residual "
            "direction in original states for a targeted analytical edit.\n")
```
