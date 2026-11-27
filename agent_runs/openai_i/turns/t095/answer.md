Changing units and pooling did not resolve the mismatch, so that explanation is not supported. The empty encodings do match. I’ll check cached records, numeric specifications, and the exposed-state layout next. If none explains the error, I’ll avoid claiming exact reconstruction and rely on reversible edits checked by the forward tool.

```python
fresh87 = pd.read_csv("/data/train.csv", nrows=20000)
cached87 = train.iloc[:len(fresh87)].reset_index(drop=True)
print("=== RECORD INTEGRITY ===")
for c87 in fresh87.columns:
    a87, b87 = fresh87[c87], cached87[c87]
    same87 = (a87 == b87) | (a87.isna() & b87.isna())
    print(c87, "differences:", int((~same87).sum()))
    if (~same87).any():
        print(pd.DataFrame({"file": a87[~same87], "cached": b87[~same87]}).head(3))

print("\n=== STATE LAYOUT ===")
outcheck87 = repair_norm(X84[:, 192:960].reshape(-1, 4, 192).mean(1), "core.out_norm")
print("out versus normalized mean(h):",
      float(((outcheck87-X84[:, :192])**2).mean().sqrt()))
print("flow.a quantiles:", torch.quantile(
    repair_weights["core.flow.a"], torch.tensor([0., .25, .5, .75, 1.])).tolist())
print("norm constants:", repair_norm.__code__.co_consts)

print("\n=== RETAINED FIRST-ACTION CONTEXT ===")
for n87 in ["ctx63", "context64", "h64", "flowed64", "gap_vec64", "e64", "u64"]:
    a87 = globals().get(n87)
    if isinstance(a87, torch.Tensor):
        print(n87, tuple(a87.shape))
        if a87.shape == (1, 192) and n87 in ["ctx63", "context64"]:
            print("  RMS versus rebuilt context[3]:",
                  float(((a87[0]-C84[3])**2).mean().sqrt()))
        if a87.shape == (1, 4, 192) and n87 == "h64":
            print("  RMS versus preceding exposed h:",
                  float(((a87[0]-X84[2, 192:960].reshape(4, 192))**2).mean().sqrt()))

print("\n=== NUMERIC SPECIFICATION INTEGRITY ===")
print("VOCAB numeric type:", type(VOCAB["numeric"]).__name__)
nv87 = VOCAB["numeric"]
if isinstance(nv87, list):
    print("first specification:", nv87[0])
elif isinstance(nv87, dict):
    print("first entry:", next(iter(nv87.items())))
print("current finger specification:", numeric_specs[("glucose_finger", "value")])
print("atom-base constants:", str(repair_atom_base.__code__.co_consts)[:1000])

print("\n=== EARLIER EXACT-CASE OBJECTS ===")
for n87, a87 in sorted(list(globals().items())):
    if n87.endswith("59") and any(s in n87 for s in ["case", "ctx", "gap", "target"]):
        if isinstance(a87, (np.ndarray, torch.Tensor)):
            print(n87, tuple(a87.shape))
        elif isinstance(a87, list):
            print(n87, "length", len(a87), "first", str(a87[:1])[:400])
print("Current edits:", edited())

with open("/work/notes.md", "a") as f87:
    f87.write("\n## Python 87; forward 6 — integrity and state-layout checks\n"
              "- Unit/pooling changes did not remove reading-feature error; that explanation is unsupported.\n"
              "- Empty observation/action vectors agree with earlier reconstruction to about 2e-7 RMS.\n"
              "- Checking cached records, numeric specifications, and h/out layout before further reconstruction work.\n")
```
