The first probes failed: they recovered broad dosing patterns, but not deviations from the dosing policy on held-out stays. That weakens the case for the simple head edit I proposed. I’ll now separate two possible causes—loss of dose information in the state versus loss in the reading head—and remove the event-specific gating from the probe. Recovery in an earlier representation would identify a better edit location.

```python
# Recover the same residual-dose targets for every reading, without asking a
# single linear probe to gate itself separately for glucose and MAP.
residual_full = np.zeros((len(head_meta), 2), dtype=np.float32)
rawdose_full = np.zeros_like(residual_full)
for i, row in enumerate(head_meta.itertuples()):
    interval = int(round(row.hour)) - 1
    if interval < 0 or (row.stay_id, interval) not in tpanel_lookup.index:
        continue
    pr = tpanel_lookup.loc[(row.stay_id, interval)]
    for j, drug_name in enumerate(["insulin", "norepinephrine"]):
        residual_full[i, j] = pr[drug_name + "_residual"]
        rawdose_full[i, j] = pr[drug_name]

reprs = {
    "query": q_head.numpy(),
    "head_input": head_input.numpy(),
    "head_hidden": head_hidden.numpy(),
    "out": state_x[:, :192].numpy(),
    "mean_h": state_x[:, 192:960].reshape(-1, 4, 192).mean(1).numpy(),
    "z_posterior": state_x[:, 960:].numpy(),
    "all_state": state_x.numpy()
}
valid_probe = head_meta.hour.to_numpy() >= 1
ungated_probe_results = []
for label0, xx in reprs.items():
    scale0 = np.maximum(xx.std(0), .03)
    zz = xx / scale0
    for alpha0 in [1., 30., 300.]:
        pp = np.zeros_like(residual_full)
        pp_raw = np.zeros_like(rawdose_full)
        for fold0 in [0, 1]:
            tr0 = (probe_fold != fold0) & valid_probe
            te0 = (probe_fold == fold0) & valid_probe
            pp[te0] = Ridge(alpha=alpha0).fit(zz[tr0], residual_full[tr0]).predict(zz[te0])
            pp_raw[te0] = Ridge(alpha=alpha0).fit(zz[tr0], rawdose_full[tr0]).predict(zz[te0])
        ungated_probe_results.append({
            "representation": label0, "alpha": alpha0,
            "residual_R2": np.round([
                r2_score(residual_full[valid_probe, j], pp[valid_probe, j]) for j in range(2)], 3).tolist(),
            "raw_R2": np.round([
                r2_score(rawdose_full[valid_probe, j], pp_raw[valid_probe, j]) for j in range(2)], 3).tolist()
        })
print("=== UNGATED, HELD-OUT-STAY PROBES ===")
for row in ungated_probe_results:
    print(row)

print("\n=== ATOM EMBEDDING TABLES ===")
for name in NAMES:
    if name.startswith(("embed.", "encode.")):
        print(name, SHAPES[name])

# Check whether the action-time posterior already contains residual-dose information.
action_mask = (state_df.hour.to_numpy() > 0) & (
    state_df.has_insulin.to_numpy() | state_df.has_norepinephrine.to_numpy())
action_rows = np.flatnonzero(action_mask)
action_targets = np.zeros((len(action_rows), 2), dtype=np.float32)
action_raw = np.zeros_like(action_targets)
for i, rindex in enumerate(action_rows):
    row = state_df.iloc[rindex]
    interval = int(np.floor(row.hour + 1e-5))
    if (row.stay_id, interval) in tpanel_lookup.index:
        pr = tpanel_lookup.loc[(row.stay_id, interval)]
        for j, drug_name in enumerate(["insulin", "norepinephrine"]):
            action_targets[i, j] = pr[drug_name + "_residual"]
            action_raw[i, j] = pr[drug_name]
action_fold = (state_df.iloc[action_rows].stay_id.to_numpy() % 2).astype(int)
for label0, xx in {
    "post_z_at_action": S0["x"][action_rows, 960:],
    "out_at_action": S0["x"][action_rows, :192],
    "all_at_action": S0["x"][action_rows]
}.items():
    zz = xx / np.maximum(xx.std(0), .03)
    pp = np.zeros_like(action_targets)
    for fold0 in [0, 1]:
        tr0, te0 = action_fold != fold0, action_fold == fold0
        pp[te0] = Ridge(alpha=30.).fit(zz[tr0], action_targets[tr0]).predict(zz[te0])
    print(label0, "residual R2",
          [round(r2_score(action_targets[:, j], pp[:, j]), 3) for j in range(2)])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 13: test earlier representations\n"
            "Initial event-gated head probes failed held-out residual-dose recovery. "
            "Testing ungated probes across query, hidden head, full state, and posterior "
            "to locate the bottleneck; inspected atom encoder tables.\n"
            f"Ungated results: {ungated_probe_results}\n"
            "Python runs: 13. Forward requests: 3.\n")
```