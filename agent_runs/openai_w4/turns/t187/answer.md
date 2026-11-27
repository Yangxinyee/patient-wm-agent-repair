The retained tests show a stable response direction with an immediate insulin effect near −1.9 per rate unit at scale 1. Its response fades when later readings are held factual, but that does not rule out sustained effects in free rollout: factual readings interrupt the feedback path. I’ll compare stable strengths using the same one-dose increment, then validate promising candidates with the actual rollout.

```python
def probabilities166(x):
    result = head_features_fixed(
        dict(reference_bank125, x=x), dataset159, draws=4, seed=2501
    )
    options = result if isinstance(result, (tuple, list)) else [result]
    arrays = []
    for a in options:
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        if isinstance(a, np.ndarray) and a.shape == (len(dataset159), 17):
            arrays.append(a)
    assert len(arrays) == 1, [
        (type(a).__name__, getattr(a, "shape", None)) for a in options
    ]
    return arrays[0]

print("Dataset columns:", list(dataset159.columns))
print("Centre objects:", type(centres159).__name__,
      getattr(centres159, "shape", None), type(centres_for159).__name__,
      getattr(centres_for159, "shape", None))

centres166 = None
for c166 in [centres159, centres_for159]:
    if torch.is_tensor(c166):
        c166 = c166.detach().cpu().numpy()
    if isinstance(c166, np.ndarray) and c166.shape == (len(dataset159), 17):
        centres166 = c166
        break
if centres166 is None:
    vc166 = VOCAB["value_bin_centres"]
    def centre_row166(ev):
        cid = code_ids_repair[ev]
        if isinstance(vc166, dict):
            for key in [ev, str(cid), cid]:
                if key in vc166:
                    return np.asarray(vc166[key], dtype=float)
            raise KeyError(ev)
        return np.asarray(vc166[cid], dtype=float)
    centres166 = np.stack([centre_row166(ev) for ev in dataset159.event])
assert centres166.shape == (len(dataset159), 17)

saved_weights166 = {n: torch.as_tensor(load(n)).clone() for n in NAMES}
outer166 = torch.outer(direction161, decoder160)
difference166 = weights161["core.b_in.weight"] - channel_weights160["core.b_in.weight"]
recovered_scale166 = float((difference166 * outer166).sum() / outer166.square().sum())
residual166 = float(torch.linalg.vector_norm(
    difference166 - recovered_scale166 * outer166
) / torch.linalg.vector_norm(difference166))
print("Recovered candidate scale:", recovered_scale166, "relative residual:", residual166)
assert residual166 < 1e-4

selected166 = impulse_results159["insulin"]["chosen"].copy()
indices166 = selected166.bundle_index.to_numpy(dtype=int)
rates166 = selected166.rate.to_numpy(dtype=float)
doses166 = selected166.dose.to_numpy(dtype=float)
increment166 = 0.5
encoded_base166 = rp_action_encoder(atom_exact65(
    "insulin", np.full(len(selected166), np.nan), rates166, doses166))
encoded_plus166 = rp_action_encoder(atom_exact65(
    "insulin", np.full(len(selected166), np.nan),
    rates166 + increment166, doses166 + increment166))
parity166 = float(torch.max(torch.abs(encoded_base166 - saved_u160[indices166])))
print("Action-encoding parity max error:", parity166)
assert parity166 < 2e-5

hour_by_stay166 = selected166.set_index("stay_id").hour.map(np.floor)
lags166 = dataset159.hour.to_numpy() - dataset159.stay_id.map(hour_by_stay166).to_numpy()
old_U166 = U65
impulse_tests166 = {}
try:
    for label166 in ["submitted", 1.0, 2.0, 2.5]:
        candidate166 = dict(saved_weights166 if label166 == "submitted"
                            else channel_weights160)
        if label166 != "submitted":
            candidate166["core.b_in.weight"] = (
                channel_weights160["core.b_in.weight"] + label166 * outer166)
        U65 = saved_u160.clone()
        xb166, qb166 = replay_factual125(candidate166)
        pb166 = probabilities166(xb166)
        U65 = saved_u160.clone()
        U65[indices166] = encoded_plus166
        xp166, qp166 = replay_factual125(candidate166)
        pp166 = probabilities166(xp166)
        response166 = ((pp166 - pb166) * centres166).sum(axis=1) / increment166
        summary166 = {}
        for ev166 in ["glucose_finger", "map_arterial"]:
            summary166[ev166] = [
                float(np.mean(response166[
                    (dataset159.event.to_numpy() == ev166) &
                    np.isclose(lags166, lag166)]))
                if np.any((dataset159.event.to_numpy() == ev166) &
                          np.isclose(lags166, lag166)) else None
                for lag166 in range(1, 7)
            ]
        impulse_tests166[label166] = summary166
        print(label166, summary166)
finally:
    U65 = old_U166

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 166\nReconstructed the ridge-0.1 connection direction exactly "
            "and tested a +0.5 insulin impulse at submitted/1/2/2.5 strengths. "
            "Later readings remain factual: fading responses do not refute sustained "
            "free-rollout effects mediated by generated readings. Encoding parity "
            "checked; results retained in impulse_tests166. No saved weights changed.\n")
```