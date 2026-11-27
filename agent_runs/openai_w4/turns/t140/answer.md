The rejected connection is removed. The CPU transition reads the preceding bundle’s state, so it should support full factual replay. I’ll first check replay against exported original states, then compare the channel-only and rejected edits. Close baseline agreement is required before trusting any diagnosis of feedback or persistence.

```python
import time

reference_bank125 = {
    k: np.asarray(v).copy() if isinstance(v, (np.ndarray, torch.Tensor)) else v
    for k, v in repair_bank.items()
}
reference_x125 = reference_bank125["x"].copy()
stay_sequences125 = [
    np.flatnonzero(reference_bank125["stay_id"] == s)
    for s in np.unique(reference_bank125["stay_id"])
]
assert all(np.all(np.diff(reference_bank125["hour"][s]) >= 0)
           for s in stay_sequences125)
replay_levels125 = [
    np.array([s[j] for s in stay_sequences125 if len(s) > j], dtype=int)
    for j in range(1, max(map(len, stay_sequences125)))
]

def replay_factual125(weights=None, capture_decoder=True):
    global repair_bank, rW, rp_linear
    old_bank, old_weights, old_linear = repair_bank, rW, rp_linear
    local_bank = dict(reference_bank125)
    local_bank["x"] = reference_x125.copy()
    local_weights = dict(old_weights if weights is None else weights)
    q_capture = np.full((len(reference_x125), 4), np.nan, dtype=np.float32)
    active_indices = None

    def replay_linear(x, prefix):
        if capture_decoder and prefix == "core.b_in":
            q_capture[active_indices] = (
                x.detach().cpu().numpy() @ connection_decoder123
            ).astype(np.float32)
        return F.linear(x, local_weights[prefix + ".weight"],
                        local_weights.get(prefix + ".bias"))

    try:
        repair_bank, rW, rp_linear = local_bank, local_weights, replay_linear
        for active_indices in replay_levels125:
            result = transition73(active_indices)
            local_bank["x"][active_indices] = result.detach().cpu().numpy()
        return local_bank["x"].copy(), q_capture
    finally:
        repair_bank, rW, rp_linear = old_bank, old_weights, old_linear

weights_original125 = {
    n: torch.as_tensor(load_original(n), dtype=torch.float32) for n in NAMES
}
weights_channel125 = {
    n: torch.as_tensor(load(n), dtype=torch.float32) for n in NAMES
}
weights_rejected125 = dict(weights_channel125)
weights_rejected125["core.b_in.weight"] = torch.as_tensor(
    connection_weight123, dtype=torch.float32
)

t125 = time.time()
replayed_original125, unused_q125 = replay_factual125(
    weights_original125, capture_decoder=False
)
original_rmse125 = np.sqrt(np.mean(
    (replayed_original125[:, :960] - reference_x125[:, :960]) ** 2
))
print("Original complete-replay RMSE:", original_rmse125)
print("Elapsed seconds:", round(time.time() - t125, 2))

if original_rmse125 < 0.02:
    replayed_channel125, q_channel125 = replay_factual125(weights_channel125)
    replayed_rejected125, q_rejected125 = replay_factual125(weights_rejected125)
    for label125, states125, qs125 in [
        ("channel-only", replayed_channel125, q_channel125),
        ("rejected connection", replayed_rejected125, q_rejected125),
    ]:
        print("\n", label125)
        for lo125, hi125 in [(-2, 6), (6, 12), (12, 18), (18, 25)]:
            mask125 = ((reference_bank125["hour"] >= lo125) &
                       (reference_bank125["hour"] < hi125))
            dq125 = np.nanmean(qs125[mask125], axis=1)
            print(
                (lo125, hi125),
                "out RMSE", round(float(np.sqrt(np.mean(
                    (states125[mask125, :192] - reference_x125[mask125, :192]) ** 2
                ))), 5),
                "h RMSE", round(float(np.sqrt(np.mean(
                    (states125[mask125, 192:960] - reference_x125[mask125, 192:960]) ** 2
                ))), 5),
                "q quantiles", np.nanquantile(dq125, [.01, .5, .99]).round(4),
            )
    print("\nTotal elapsed seconds:", round(time.time() - t125, 2))
else:
    print("Replay validation failed; do not use its candidate diagnostics.")

print("\nReading-record DataFrames:")
for n125, v125 in list(globals().items()):
    if isinstance(v125, pd.DataFrame) and "event" in v125.columns:
        print(n125, v125.shape)

with open("/work/notes.md", "a") as f125:
    f125.write(
        "\n## Run 125\n"
        f"Implemented replay_factual125 using stay-separated sequential updates "
        f"of repair_bank['x']; original full-state RMSE={original_rmse125:.6g}. "
        "Retained weights_channel125, weights_rejected125 and replay states/q "
        "captures for dynamic diagnosis. No additional saved edits.\n"
    )
```
