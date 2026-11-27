The rollout rejects the current calibration: insulin responses became smaller, and doubled norepinephrine predicted a MAP decrease. Factual preservation alone is not enough. I’ll restore the original weights and separate the two drug corrections so they can be calibrated independently; opposite-signed corrections improving both dose directions would support a routing error, while continued inconsistency would require a different pathway.

```python
for name126 in list(edited()):
    drop_edit(name126)
print("Restored edits:", edited())

base126 = candidates90[0.0]
one126 = candidates90[1.0]
print("\n=== CANDIDATE LINEARITY ===")
for name126 in base126:
    b126 = np.asarray(base126[name126], dtype=np.float64)
    d126 = np.asarray(one126[name126], dtype=np.float64) - b126
    e126 = np.asarray(candidates90[2.0][name126], dtype=np.float64) - b126 - 2*d126
    print(name126, "delta norm", round(float(np.linalg.norm(d126)), 6),
          "scale-2 nonlinearity", round(float(np.linalg.norm(e126)), 6))

dw126 = np.asarray(one126["z_post.2.weight"]) - np.asarray(base126["z_post.2.weight"])
db126 = np.asarray(one126["z_post.2.bias"]) - np.asarray(base126["z_post.2.bias"])
deltaM126 = np.vstack([dw126.T, db126[None, :]])
K126 = np.linalg.lstsq(np.asarray(Binnov89), deltaM126, rcond=None)[0]
drug_parts126 = [
    np.outer(np.asarray(Binnov89)[:, j], K126[j]) for j in range(2)
]
re126 = np.linalg.norm(sum(drug_parts126) - deltaM126) / max(np.linalg.norm(deltaM126), 1e-12)
print("Drug-separated reconstruction relative error:", float(re126))
print("Drug template norms:", [float(np.linalg.norm(x)) for x in drug_parts126])
print("Readout file columns:", pd.read_csv("/work/fwd_6/dose_edit_original.csv", nrows=2).columns.tolist())

print("\n=== STORED DYNAMIC TARGET EVIDENCE ===")
lines126 = [x for x in Path("/work/notes.md").read_text().splitlines()
            if re.search(r"aggregate|rho|half.life|cumulative|rate.unit|bootstrap|impulse estimates|dynamic estimates", x, re.I)]
print("\n".join(lines126[:22])[:2600])
print("\n=== RETAINED PANEL OBJECTS ===")
for k, v in list(globals().items()):
    if isinstance(v, pd.DataFrame) and len(v) > 10000:
        print(k, v.shape, v.columns.tolist()[:12])
print("\n=== SMALL DYNAMIC OBJECTS ===")
for k, v in list(globals().items()):
    if re.search(r"impulse|rho|gamma|dynamic", k, re.I):
        if isinstance(v, dict):
            print(k, list(v.keys())[:8])
        elif isinstance(v, np.ndarray):
            print(k, v.shape, str(v)[:200] if v.size < 30 else "")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 126: official rejection and drug-separated calibration\n"
            "- Forward 5 factual: original 1.63155, candidate 1.63154; preservation passed on tuning-100.\n"
            "- Forward 6 rollout candidate: insulin x2 -1.242, half +0.495; NE x2 -0.666, half -0.395.\n"
            "- Candidate response is worse than original and reverses NE doubling; restored all original weights.\n"
            "- Decomposing token correction by drug for independent official calibration; local response signs are not reliable.\n"
            "- Python runs: 126; forward passes: 6.\n")
```
