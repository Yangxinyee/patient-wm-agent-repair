The fully balanced insulin trial was also stronger than the baseline, so its failure does not rule out a small, strength-matched correction. I’ll install a 15% blend toward its balanced detector and test factual fit first. It preserves the intended mean ×2 sensitivity while increasing ×½ sensitivity; the GPU check will determine whether that benefit is affordable.

```python
print("=== MEASURED FACTUAL FRONTIER ===")
for k219 in range(1, 24):
    p219 = "/work/fwd_%d/summary.json" % k219
    if not os.path.exists(p219):
        continue
    sm219 = json.load(open(p219))
    if sm219.get("type") == "factual":
        rr219 = sm219["results"]
        if "original" in rr219 and "edited" in rr219:
            oo219, ee219 = rr219["original"], rr219["edited"]
            print(k219, "stays", sm219["n_stays"],
                  "NLL_delta", round(ee219["value_nll_per_value"] -
                                     oo219["value_nll_per_value"], 6),
                  "event_deltas",
                  {c: round(ee219["by_event"][c]["nll"] -
                            oo219["by_event"][c]["nll"], 5)
                   for c in oo219["by_event"]})

print("\n=== STAY-WEIGHTED ROLLOUT FRONTIER ===")
for k219 in range(1, 24):
    p219 = "/work/fwd_%d/dose_edit_edited.csv" % k219
    if os.path.exists(p219):
        dd219 = pd.read_csv(p219)
        if k219 == 3:
            print("columns", list(dd219.columns))
        editcol219 = next((c for c in ["edit", "dose_edit", "intervention"]
                           if c in dd219.columns), None)
        staycol219 = next((c for c in ["stay_id", "stay"]
                           if c in dd219.columns), None)
        if editcol219 and staycol219 and "shift" in dd219.columns:
            vals219 = dd219.groupby([editcol219, staycol219])["shift"].mean()
            print(k219, vals219.groupby(level=0).mean().round(3).to_dict())

a58_219 = np.load("/work/insulin_candidate58.npz")
a63_219 = np.load("/work/insulin_candidate63.npz")
base219 = np.load("/work/baseline77.npz")["delta"].astype(np.float64)
balanced219 = a63_219["delta"].astype(np.float64) * (1.6 / 1.9)
blend219 = 0.15
delta219 = base219 + blend219 * (balanced219 - base219)
fbase219 = 1.5 * 1.6 * a58_219["input_feature"]
fbalanced219 = 1.5 * 1.6 * a63_219["feature"]
ftrial219 = (1 - blend219) * fbase219 + blend219 * fbalanced219
direction219 = a58_219["direction"]

print("\n=== PARTIAL-BALANCE AUDIT ===")
print("baseline reconstruction max error:",
      float(np.max(np.abs(base219 - np.outer(direction219, fbase219)))))
print("balanced reconstruction max error:",
      float(np.max(np.abs(balanced219 -
                          np.outer(direction219, fbalanced219)))))
print("trial delta norm:", float(np.linalg.norm(delta219)))
guard219 = float(np.max(np.abs(initial_matrix56 @ ftrial219)))
print("initial admission guard max:", guard219)
for label219, ff219 in [("baseline", fbase219), ("partial", ftrial219)]:
    try:
        print(label219, fun77(ff219))
    except Exception as exc219:
        print(label219, "diagnostic unavailable:", str(exc219)[:150])

orig219 = load_original("core.b_in.weight")
origarr219 = (orig219.detach().cpu().numpy() if torch.is_tensor(orig219)
              else np.asarray(orig219))
if guard219 < 1e-3:
    w219 = (origarr219 + delta219).astype(np.float32)
    save_edit("core.b_in.weight",
              torch.from_numpy(w219) if torch.is_tensor(orig219) else w219)
    np.savez("/work/insulin_trial219.npz", delta=delta219,
             blend=blend219, feature=ftrial219, direction=direction219)
    print("Installed reversible partial-balance trial.")
else:
    print("NOT INSTALLED: admission guard failed.")

fact219 = pd.read_csv("/work/fwd_21/factual_original.csv")
staycol219 = next(c for c in ["stay_id", "stay"] if c in fact219.columns)
ids219 = fact219[staycol219].drop_duplicates().astype(int).tolist()
print("\nFACTUAL_VALIDATION_IDS =", ids219)

with open("/work/notes.md", "a") as f219:
    f219.write(
        "\n## Run 219: partial insulin balancing trial\n"
        "Installed baseline77 + .15*(candidate63 delta*1.6/1.9 - baseline77), "
        "matching original baseline strength rather than using rejected "
        "candidate63 strength1.9. Intended x2 mean detector sensitivity is "
        "unchanged; half-dose sensitivity increases. Exact initial admission "
        "guard checked. Unaccepted pending GPU factual and rollout tests. "
        "Archived insulin_trial219.npz. Python runs: 219; forwards: 23.\n"
    )
```
