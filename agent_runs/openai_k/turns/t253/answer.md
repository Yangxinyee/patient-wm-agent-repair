The recovered inputs have the expected tensor shapes. The failing helper substitutes a latent token into the replay, so I need to check whether that substitution—not the recurrent computation—caused the tuple error. I’ll also recover the head calculation and earlier effect estimates. A validated replay would support testing a correction later in the recurrent update; otherwise it would not.

```python
import dis, collections

def compact227(f):
    lines = collections.defaultdict(list)
    line = None
    for ins in dis.get_instructions(f):
        if ins.starts_line is not None:
            line = ins.starts_line
        lines[line].append(ins.opname.replace("LOAD_", "L_").replace("STORE_", "S_")
                           + (" " + ins.argrepr if ins.argrepr else ""))
    return "\n".join(str(k) + ": " + "; ".join(v) for k, v in lines.items())

for name227 in ("repaired_core_local", "local_token154",
                "r49_readouts", "r179_capture_head"):
    f227 = globals().get(name227)
    if callable(f227):
        print("\n==", name227, "==")
        print(compact227(f227)[:2100 if name227 != "repaired_core_local" else 3100])

print("\n=== STORED SOURCE ===")
for k227 in ("rr33_source", "sources116"):
    if k227 in globals():
        o227 = globals()[k227]
        print(k227, type(o227).__name__,
              len(o227) if hasattr(o227, "__len__") else "",
              list(o227.keys())[:12] if isinstance(o227, dict) else "")

print("=== EARLY RECORD-EVIDENCE NOTES ===")
text227 = open("/work/notes.md").read()
paras227 = text227.split("\n\n")
selected227 = [p for p in paras227
               if any(s in p.lower() for s in
                      ("instrument", "cross-fit", "crossfit", "impulse",
                       "lag coefficients", "autoregress", "identif"))
               and not p.lstrip().startswith("## Run 2")]
print("\n\n".join(selected227[:6])[:1600])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 227: compact core and head recovery\n"
            "- Confirmed retained replay context contains correctly shaped tensors.\n"
            "- Recovering compact instructions for token substitution, recurrent update and value readout.\n"
            "- No weights changed. Python runs: 227; forward passes: 17.\n")
```