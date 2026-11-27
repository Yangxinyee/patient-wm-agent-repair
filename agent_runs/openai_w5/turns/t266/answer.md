The records support negative insulin effects and positive norepinephrine effects, though the long-run targets still depend on the fitted response model. The present edit improves doubling responses but underpredicts halving responses. I’ll inspect its sensor-specific controls and saved weights so I can test strength and centering separately on the GPU; a shared scaling fix would support simple undercalibration, while persistent asymmetry would refute it.

```python
def summarize234(x, depth=0):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        s = f"{type(x).__name__}{a.shape}"
        if a.size <= 18:
            s += " " + repr(a.tolist())
        return s
    if isinstance(x, dict):
        if depth >= 2:
            return "dict keys=" + repr(list(x)[:25])
        return {str(k): summarize234(v, depth + 1) for k, v in list(x.items())[:20]}
    if isinstance(x, (list, tuple)):
        if len(x) <= 12 and depth < 2:
            return [summarize234(v, depth + 1) for v in x]
        return f"{type(x).__name__}[{len(x)}]"
    if callable(x):
        return "callable " + getattr(x, "__name__", "")
    return repr(x)[:180]

print("=== CONTROL / CHECKPOINT OBJECTS ===")
for k in ["BASE217", "_controls", "_control", "_params216", "_pair_for_control",
          "_groups215", "_spec", "groups230", "head225", "head_rows",
          "head_audit", "head_q", "head_wc", "head_vb", "head_prev",
          "A", "B", "C", "F", "S", "V"]:
    if k in globals():
        print(k, summarize234(globals()[k]))

print("=== OTHER RELEVANT OBJECT NAMES ===")
print(" ".join(k for k in sorted(globals())
               if any(t in k.lower() for t in
                      ["strength", "coef", "center", "resid", "gate", "candidate",
                       "panel", "target", "dose", "sensor", "split", "tune"])))

print("=== VALUE-HEAD WEIGHT DIFFERENCES ===")
for nm in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
           "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]:
    cur = np.asarray(load(nm))
    old = np.asarray(load_original(nm))
    print(nm, cur.shape, "delta norm", float(np.linalg.norm(cur-old)))
    if cur.ndim == 2:
        axis = 1 if nm.endswith(".0.weight") else 0
        energy = np.linalg.norm(cur-old, axis=axis)
        ind = np.argsort(energy)[-16:][::-1]
        print("largest changed hidden indices:", list(zip(ind.tolist(), energy[ind].round(4).tolist())))

notes = open("/work/notes.md").read()
start = notes.find("## Run 118")
stop = notes.find("## Run 135")
print("=== TRANSPORT DESIGN ===")
print(notes[start:stop][:2400])
print("=== EARLY SENSOR CHECKS ===")
print(notes[2800:4200])
```
