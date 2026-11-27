The existing norepinephrine feature is highly asymmetric: it detects doubling but barely detects halving. Reusing it unchanged would likely miss one readout. I’ll verify the feature scaling and validation split, then fit a balanced detector for both dose directions. Similar signed sensitivity in both directions, with low factual leakage, would support proceeding.

```python
import json
import pandas as pd

def array80(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== RAW/NORMALIZED FEATURE CHECK ===")
coef80 = constrained_fits57[("zero", 10.0)]
for name80 in ["slot_features47", "slot_factual47", "prelude_all46",
               "initial_matrix56", "null57", "fit_rows57", "fit_drug57",
               "test_cf46", "relevant46", "src45", "dt45", "static45",
               "trans45", "centres45", "hnew46"]:
    obj80 = globals()[name80]
    a80 = array80(obj80)
    print(name80, a80.shape, a80.dtype,
          "range", (float(np.min(a80)), float(np.max(a80))))
for name80 in ["slot_features47", "slot_factual47", "prelude_all46"]:
    a80 = array80(globals()[name80])
    p80 = a80 @ coef80
    print(name80, "heldout RMS by output",
          np.sqrt(np.mean(p80[~fit_rows57]**2, axis=(0, 1))).tolist())
print("initial null errors:",
      float(np.max(np.abs(initial_matrix56 @ coef80))),
      float(np.max(np.abs(initial_matrix56 @ null57))),
      float(np.max(np.abs((initial_matrix56 / slot_scale47) @ null57))))
print("candidate raw equals insulin fit:",
      np.linalg.norm(candidate_raw58 - coef80[:, 0]))
print("candidate input/raw ratio:",
      np.dot(candidate_input58, candidate_raw58) /
      np.dot(candidate_raw58, candidate_raw58))

print("\n=== CONTRAST UNIT CHECK ===")
for key80 in slot_deltas47:
    d80 = array80(slot_deltas47[key80])
    y80 = array80(targets57[key80])[:, 1]
    test80 = (~fit_drug57) & (np.abs(y80) > 1e-8)
    if key80[0] == "norepinephrine":
        print(key80, "n", int(test80.sum()))
        for label80, multiplier80 in [
            ("direct", np.ones(192)),
            ("divide_scale", 1 / slot_scale47),
            ("multiply_scale", slot_scale47)
        ]:
            p80 = (d80 * multiplier80) @ coef80[:, 1]
            p80 = p80.mean(axis=1)
            slope80 = np.dot(p80[test80], y80[test80]) / np.dot(y80[test80], y80[test80])
            print(" ", label80, "slope", float(slope80))
    if key80 == ("norepinephrine", 0.5):
        print("last raw delta comparisons:",
              np.linalg.norm(d80 - array80(delta47)),
              np.linalg.norm(d80 - array80(delta47) / slot_scale47))

print("\n=== LOCAL READING CALIBRATION ===")
h80, o80 = coda46(hnew46)
m80 = array80(next_means45(h80, o80))
print("means:", m80.shape, np.round(m80, 3).tolist())
print("relevant46:", array80(relevant46).tolist())
print("src45:", array80(src45).tolist())
print("meta46 columns:", meta46.columns.tolist())
print("meta46 sample:", meta46.head(3).to_dict("records"))
print("ordered46 columns:", ordered46.columns.tolist())

print("\n=== BASELINE VALIDATION FILES ===")
for path80 in sorted(Path("/work/fwd_10").glob("*")):
    print(path80.name)
    if path80.suffix == ".csv":
        table80 = pd.read_csv(path80)
        print(" columns:", table80.columns.tolist(), "shape:", table80.shape)
        print(table80.head(2).to_dict("records"))
        if "stay_id" in table80:
            validation_ids80 = sorted(table80.stay_id.unique().tolist())
            print(" validation IDs:", validation_ids80)

with open("/work/notes.md", "a") as file80:
    file80.write(
        "\n## Run 80: balanced NE detector setup\n"
        "Existing factual-zero NE feature is strongly asymmetric between doubling "
        "and halving. Auditing raw feature units and held-out stay masks before "
        "a separately balanced closed-form fit. No weights changed. "
        "Python runs: 80; forwards: 10.\n"
    )
```