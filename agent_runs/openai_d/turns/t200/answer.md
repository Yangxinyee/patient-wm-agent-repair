Forward 22 passes the factual constraint on its 100 stays: the NLL increase is about 0.00385 nats. That leaves only a modest margin. I’ll retain this combined edit as a fallback and inspect the norepinephrine correction and target calculation before testing stronger output; amplification must improve the rollouts without crossing the factual limit.

```python
print("===== NOREPINEPHRINE IMPULSE EVIDENCE =====")
a = r171_notes.find("## Run 5")
b = r171_notes.find('"norepinephrine"', a)
c = r171_notes.find("## Run 6", b)
print(r171_notes[b:min(c if c >= 0 else b + 3000, b + 3000)])

print("\n===== EDIT-CONSTRUCTION NOTES =====")
for heading in ["## Run 124", "## Run 130", "## Run 136", "## Run 137",
                "## Run 138", "## Run 139"]:
    a = r171_notes.find(heading)
    if a >= 0:
        b = r171_notes.find("\n## ", a + 5)
        print(r171_notes[a:b if b >= 0 else a + 1000][:1000])

print("\n===== TARGET/KERNEL VARIABLES =====")
for n in sorted(list(globals())):
    if any(s in n.lower() for s in ["kernel", "target", "impulse"]):
        v = globals()[n]
        if isinstance(v, dict):
            print(n, "keys", list(v)[:15])
        elif hasattr(v, "shape"):
            print(n, tuple(v.shape), repr(v)[:150] if np.prod(v.shape) < 30 else "")
        elif not callable(v):
            print(n, repr(v)[:180])

print("\n===== NE PATCH SUPPORT =====")
for n, v in r170_ne_patch.items():
    x = torch.as_tensor(v)
    print(n, tuple(x.shape), "nonzero", int((x != 0).sum()),
          "norm", float(x.norm()))
print("NE rows:", r170_rows)
for n in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
          "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]:
    x = torch.as_tensor(load(n))
    if n.endswith("2.weight"):
        print("Current NE output columns:\n", x[:, r170_rows])
    elif n.endswith("2.bias"):
        print("Current-minus-insulin output bias:\n",
              x - torch.as_tensor(r170_insulin_backup[n]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 171–173: restart audit and combined-trial results\n"
            "Re-read architecture and record schema. Forward 22 factual on tuning 300..399: "
            "original 1.64248740, combined 1.64634215, delta +0.00385475; passes this cohort. "
            "Forward 23 on tuning 0..99: insulin x2 -14.331, half +6.691; "
            "NE x2 +2.387, half -0.873. Combined edit is a provisional fallback, "
            "not yet validated across other cohorts. NE remains weaker than provisional "
            "record-derived targets, especially for half-dose. Four-tensor backup "
            "r171_combined_backup retained. No new weight change.\n")
```
