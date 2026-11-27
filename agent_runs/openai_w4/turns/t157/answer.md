The retained tests contain full trajectories and per-reading losses, but the last output did not show their numerical summaries. I’ll extract those and a compact version of the transition equations. This will distinguish a usable stronger scale from an unstable one before spending another GPU pass.

```python
import dis
from collections import defaultdict

print("=== NUMERICAL STABILITY SUMMARY ===")
ref140 = np.asarray(reference_x125)
for s140, result140 in scale_results128.items():
    x140 = np.asarray(result140[0])
    print("scale", s140,
          "out_RMSE", round(float(np.sqrt(np.mean((x140[:, :192]-ref140[:, :192])**2))), 6),
          "h_RMSE", round(float(np.sqrt(np.mean((x140[:, 192:960]-ref140[:, 192:960])**2))), 6),
          "max_abs_out", round(float(np.max(np.abs(x140[:, :192]))), 4))
print("\n=== MEAN REPLAY NLL ===")
for s140, v140 in nll_scales129.items():
    print(s140, round(float(np.mean(np.asarray(v140))), 7))
print("baseline-related names:",
      sorted(n140 for n140 in globals()
             if ("129" in n140 or "125" in n140)
             and not callable(globals()[n140])))

print("\n=== COMPACT TRANSITION BYTECODE, LINE 14 ONWARD ===")
lines140 = defaultdict(list)
line140 = None
for ins140 in dis.get_instructions(transition73):
    if ins140.starts_line is not None:
        line140 = ins140.starts_line
    if line140 is not None and line140 >= 14:
        lines140[line140].append(ins140.opname + (
            ":" + ins140.argrepr if ins140.argrepr else ""))
for line140, ops140 in lines140.items():
    print(line140, " ".join(ops140))

print("\n=== REPLAY BYTECODE ===")
lines140 = defaultdict(list)
line140 = None
for ins140 in dis.get_instructions(replay_factual125):
    if ins140.starts_line is not None:
        line140 = ins140.starts_line
    lines140[line140].append(ins140.opname + (
        ":" + ins140.argrepr if ins140.argrepr else ""))
for line140, ops140 in lines140.items():
    print(line140, " ".join(ops140))

print("\n=== CHANNEL CAPTURE ===")
print(compact139(chosen_channel116["capture"]))
print("related globals:",
      sorted(n140 for n140 in globals()
             if any(s140 in n140 for s140 in
                    ["116", "117", "118", "112", "95"])
             and not n140.startswith(("name", "value", "obj", "fn", "buf"))))

with open("/work/notes.md", "a") as f140:
    f140.write("\n## Run 140\nExtracted numerical scale stability/NLL "
               "summaries and compact transition bytecode. No weights changed.\n")
```