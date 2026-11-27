The trial improved the difficult-cohort NLL increase from about +0.00558 to +0.00510, but it still narrowly misses the limit there. The proxy improvement therefore translated only partly into factual improvement. I’ll calculate the exact paired change and recover the record-based rollout targets before choosing a safer strength. Independent NLL below the limit, with responses near those targets, would support a final candidate.

```python
import pandas as pd

fo213 = pd.read_csv("/work/fwd_26/factual_original.csv")
fe213 = pd.read_csv("/work/fwd_26/factual_edited.csv")
print("FACTUAL COLUMNS", list(fo213.columns))
nll_col213 = next(c for c in fo213.columns if "nll" in c.lower())
join_cols213 = [c for c in ["stay_id", "hour", "event", "observed"]
                if c in fo213.columns]
assert fo213[join_cols213].equals(fe213[join_cols213])
paired213 = fo213[join_cols213].copy()
paired213["delta"] = fe213[nll_col213].to_numpy() - fo213[nll_col213].to_numpy()
print("EXACT NLL DELTA", paired213["delta"].mean())
print("BY EVENT", paired213.groupby("event")["delta"].agg(["mean", "size"]))
group213 = paired213.groupby("stay_id")["delta"].agg(["sum", "size"])
cluster_se213 = np.sqrt(
    len(group213)/(len(group213)-1) *
    np.sum((group213["sum"] - paired213["delta"].mean()*group213["size"])**2)
) / len(paired213)
print("STAY-CLUSTER SE", cluster_se213)

print("\nRECORD-TARGET HELPERS")
for name213 in ["record_impulse_targets", "linear_rollout_targets",
                "fit_latent_record_model", "sensor_iv_persistence"]:
    fn213 = globals().get(name213)
    if callable(fn213):
        print(name213, inspect.signature(fn213))

print("\nEFFECT / TARGET OBJECTS")
for k, v in sorted(dict(globals()).items()):
    if any(s in k.lower() for s in
           ["rho", "latent", "impulse", "rollout_target", "target_read", "effect_est"]):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape),
                  np.asarray(v).tolist() if v.numel() <= 8 else ""
                  if isinstance(v, torch.Tensor) else
                  np.asarray(v).tolist() if v.size <= 8 else "")
        elif isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape, list(v.columns)[:8])
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:8])
        elif not callable(v):
            print(k, type(v).__name__, repr(v)[:180])

fresh_notes213 = Path("/work/notes.md").read_text()
target_lines213 = [
    line for line in fresh_notes213.splitlines()
    if any(s in line.lower() for s in
           ["rollout target", "accumulated", "persistence", "impulse",
            "record-based target", "target shifts", "target effect"])
]
print("\nTARGET NOTES\n", "\n".join(target_lines213[-32:]))
print("\nCURRENT EDIT HASH",
      hashlib.sha256(load("core.b_in.weight").numpy().tobytes()).hexdigest()[:16])
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward26 / run213: constrained trial still borderline\n"
            "Difficult-cohort factual NLL increase approximately 0.00510, "
            "improving candidate_B204 but narrowly above 0.005 locally. "
            "No further weight changes. Recovering record-derived accumulated "
            "targets before selecting strengths; direct rollout validation "
            "and independent factual coverage remain necessary.\n")
```