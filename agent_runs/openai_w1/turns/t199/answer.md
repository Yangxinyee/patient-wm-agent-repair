The state export is available, but its summary alone does not reveal whether the branch changed anything. I’ll compare posterior, slot, and output differences by bundle type, and recover the earlier causal-analysis notes. This will tell us whether increasing the gain is justified or whether the branch needs redesigning.

```python
r179_original = np.load("/work/fwd_14/states_original.npz")
r179_edited = np.load("/work/fwd_14/states_edited.npz")
assert np.array_equal(r179_original["stay_id"], r179_edited["stay_id"])
assert np.allclose(r179_original["hour"], r179_edited["hour"])
r179_delta = r179_edited["x"] - r179_original["x"]
print("=== ACTUAL FORWARD STATE DIFFERENCES ===")
for _label, _mask in [
    ("all", np.ones(len(r179_delta), dtype=bool)),
    ("insulin", r179_original["has_insulin"].astype(bool)),
    ("norepinephrine", r179_original["has_norepinephrine"].astype(bool)),
    ("neither", ~(r179_original["has_insulin"].astype(bool) |
                  r179_original["has_norepinephrine"].astype(bool)))
]:
    print(_label, "n", int(_mask.sum()))
    for _name, _sl in [("out", slice(0,192)), ("slots", slice(192,960)),
                      ("posterior", slice(960,992))]:
        _d = r179_delta[_mask, _sl]
        print(" ", _name, "RMS", float(np.sqrt(np.mean(_d**2))),
              "max_abs", float(np.max(np.abs(_d))))
print("original posterior RMS", float(np.sqrt(np.mean(r179_original["x"][:,960:992]**2))))
print("\n=== LATENT PROJECTION ===")
for _name in NAMES:
    if _name.startswith("z_in"):
        _w = load_original(_name)
        print(_name, tuple(_w.shape), "RMS", float(torch.sqrt(torch.mean(_w**2))))
        if _w.ndim == 2 and _w.shape[1] == 32:
            for _drug, _spec in rp_branch_specs.items():
                print(" ", _drug, "physical projection norm",
                      float(torch.linalg.vector_norm(_w @ torch.tensor(_spec["physical"], dtype=torch.float32))))

r179_fo = pd.read_csv("/work/fwd_12/factual_original.csv")
r179_fe = pd.read_csv("/work/fwd_12/factual_edited.csv")
print("\n=== FACTUAL CSV DIFFERENCES ===")
print("columns", list(r179_fo.columns))
for _col in ("nll", "predicted_mean"):
    if _col in r179_fo and len(r179_fo) == len(r179_fe):
        _d = r179_fe[_col].to_numpy() - r179_fo[_col].to_numpy()
        print(_col, "mean", float(np.mean(_d)), "max_abs", float(np.max(np.abs(_d))),
              "nonzero", int(np.count_nonzero(_d)))

print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text()[:3000])
r179_tuning = pd.read_csv("/data/tuning.csv")
print("tuning", r179_tuning.shape)
print(r179_tuning[r179_tuning.event.isin(["insulin","norepinephrine"])].head(8).to_string(index=False))
print("\n=== EARLY CAUSAL EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:4700])
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Forward 14 / Run 179\n"
             "Exported paired states for posterior branch gain 4/4; comparing actual latent, slot "
             "and output changes to distinguish gate inactivity from weak transmission. "
             "No new edit. Counts: 179 Python runs, 14 forwards.\n")
```
