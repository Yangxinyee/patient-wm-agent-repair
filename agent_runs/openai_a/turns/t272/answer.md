Both GPU batches show substantial factual loss on insulin-inactive finger readings, and the cached screen suggests gating can reclaim some of that budget. The fallback reconstruction also matches exactly. I will install a reversible gated scale-8 trial; it succeeds only if GPU factual loss stays below the limit and insulin rollouts improve.

```python
import hashlib

assert set(edited()) == set(stronger6_checkpoint)
for name in stronger6_checkpoint:
    assert np.array_equal(as_numpy(load(name)), as_numpy(stronger6_checkpoint[name]))

activity_gate_config = {
    "scale": 8.0, "finger_gate_strength": 30.0,
    "activity_strength": 180.0, "activity_centre": 0.05,
    "w": activity_w.copy(), "b": activity_b,
    "contrast_w": w_ins.copy(), "contrast_b": b_ins,
}
gated8_candidate = make_selective_contrast_graft(w_ins, b_ins, 8.0, 30.0)
for node in map(int, sc_nodes):
    gated8_candidate[sc_names[0]][node] += np.float32(180.0) * activity_w
    gated8_candidate[sc_names[1]][node] += np.float32(180.0 * (activity_b - 0.05))

for name, array in gated8_candidate.items():
    assert tuple(array.shape) == tuple(SHAPES[name])
    assert np.isfinite(array).all()
    save_edit(name, torch.from_numpy(np.asarray(array, dtype=np.float32).copy()))

gated8_checkpoint = {name: load(name).clone() for name in edited()}
gated8_hash = hashlib.sha256(
    b"".join(as_numpy(gated8_checkpoint[n]).tobytes() for n in sorted(gated8_checkpoint))
).hexdigest()[:12]

tuning_ids_all = np.sort(
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique()
)
print("Installed activity-gated scale 8; hash:", gated8_hash)
print("Tuning stay count:", len(tuning_ids_all))
print("First-100 IDs:", tuning_ids_all[:100].tolist())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Activity-gated scale-8 trial; Python 237, forwards 26/50\n"
            "- Exact stronger6 reconstruction verified before editing.\n"
            "- Actual scale7 finger NLL cost is substantial on inactive hours: "
            "first100 +0.00947; second100 +0.02502 nats per inactive finger reading.\n"
            "- Installed scale8, finger gate30, activity gate180 centred at +0.05; "
            "activity ridge=.003 with invariance penalty100.\n"
            "- Four value-head tensors only. Trial is unaccepted pending paired GPU tests.\n"
            "- stronger6_checkpoint remains the validated fallback.\n")
```
