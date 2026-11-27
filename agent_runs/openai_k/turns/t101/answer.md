The saved forward results confirm the discrepancy: factual NLL is 1.638 on the same training stays, versus about 5.064 locally. This is not a split difference. I’ll compare predictions for identical readings and inspect the readout contract. That should reveal whether the problem is row alignment or the reconstructed head calculation.

```python
import dis, inspect, numpy as np, pandas as pd, torch

def compact_code95(fn, limit=3600):
    lines = []
    current = []
    lineno = None
    for ins in dis.get_instructions(fn):
        if ins.starts_line is not None and current:
            lines.append(f"{lineno}: " + " | ".join(current))
            current = []
        if ins.starts_line is not None:
            lineno = ins.starts_line
        if ins.opname in {
            "RESUME", "PUSH_NULL", "PRECALL", "COPY", "SWAP", "EXTENDED_ARG",
            "GET_ITER", "END_FOR", "NOP", "POP_TOP", "LOAD_FAST_AND_CLEAR",
            "MAKE_CELL", "LOAD_CLOSURE"
        }:
            continue
        current.append(ins.opname.replace("LOAD_", "L:").replace("STORE_", "S:")
                       + (" " + ins.argrepr if ins.argrepr else ""))
    if current:
        lines.append(f"{lineno}: " + " | ".join(current))
    text = "\n".join(lines)
    print(text[:limit])
    if len(text) > limit:
        print("... omitted", len(text) - limit, "characters")

print("READOUT IMPLEMENTATION")
print("r49_readouts references:", r49_readouts.__code__.co_names)
compact_code95(r49_readouts, 4300)

print("\nFEATURE RETURN CONTRACT")
for ins95 in list(dis.get_instructions(r59_features))[-35:]:
    print(ins95.opname, ins95.argrepr)

print("\nRELEVANT ARRAY INVENTORY")
for name95 in sorted(list(globals())):
    if name95.startswith("r49") or name95.endswith("80"):
        val95 = globals()[name95]
        if isinstance(val95, (np.ndarray, torch.Tensor)):
            print(name95, tuple(val95.shape))
        elif inspect.isfunction(val95):
            print(name95, str(inspect.signature(val95)))
        elif isinstance(val95, (dict, tuple, list)):
            print(name95, type(val95).__name__, len(val95),
                  list(val95)[:5] if isinstance(val95, dict) else "")

print("\nINDEX SAMPLES")
print("state stay/hour:", list(zip(rr33_stays[rows80[:12]],
                                 rr33_hours[rows80[:12]])))
print("target index:", list(future_index59[:12]))
print("factual events at first stay:")
factual95 = pd.read_csv("/work/fwd_4/factual_original.csv")
first95 = int(rr33_stays[rows80[0]])
print(factual95[(factual95.stay_id == first95) &
                (factual95.hour.between(0, 4))].to_string(index=False))

print("\nACTIVE LOW-LEVEL FUNCTIONS")
print("mlp0", mlp0.__name__, "lin0", lin0.__name__)
print("Edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 95: same-stay factual mismatch\n"
            "- External original factual NLL on training-100 is 1.63805; local score "
            "on these stays is ~5.064. The mismatch is not explained by train/tuning.\n"
            "- Auditing readout implementation and identical-reading alignment. "
            "Local edit scores are provisional until this passes.\n"
            "- No edits. Python runs: 95; forward passes: 4.\n")
```