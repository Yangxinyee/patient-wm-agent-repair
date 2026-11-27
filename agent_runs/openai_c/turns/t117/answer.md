The latest message repeats the rollout result; the state inspection has not run yet. I’ll safely restore numeric gain 4 and inspect the saved paired states. Their alignment and dose normalisation must be verified before using them to design another edit.

```python
r117_ops = ops()
if len(r117_ops) == 3 and [r117_o["name"] for r117_o in r117_ops] == [
    "embed.numeric.weight", "embed.code.weight", "embed.code.weight"
]:
    undo()
    undo()
assert len(ops()) == 1 and ops()[0]["name"] == "embed.numeric.weight"
print("RESTORED:", [{k:v for k,v in r117_o.items() if k != "vector"}
                    for r117_o in ops()])

print("\nVOCAB KEYS:", list(VOCAB))
for r117_k in VOCAB:
    if "norm" in r117_k.lower():
        print(r117_k, str(VOCAB[r117_k])[:2400])

print("\nPAIRED STATE FILES:", os.listdir("/work/fwd_13"))
for r117_model in ("original", "edited"):
    r117_npz = np.load(f"/work/fwd_13/states_{r117_model}.npz")
    print(r117_model, [(r117_k, r117_npz[r117_k].shape)
                      for r117_k in r117_npz.files])
    for r117_k in r117_npz.files:
        if r117_npz[r117_k].ndim == 1:
            print(r117_k, r117_npz[r117_k][:4])

print("\nNUMERIC DOSE DIRECTION:", dose_numeric_direction)
print("CONTROL COLUMNS:", list(sc.columns))
print("PAIRED ARRAY SHAPES:",
      {r117_n: globals()[r117_n].shape
       for r117_n in ("xo", "xe", "delta_state", "before")})

with open("/work/notes.md", "a") as r117_f:
    r117_f.write(
        "\n## Forward 32 / Python run 82\n"
        "- Full identity removal: insulin -2.232/+1.670, NE +0.203/-0.273; rejected.\n"
        "- Numeric gain 4 alone restored. Inspecting paired controlled state perturbations.\n"
        "- Forward passes used: 32; 18 remain.\n"
    )
```