The records show persistent glucose effects and faster-decaying MAP effects, so the next-hour slopes alone understate the rollout targets. I’ll convolve those estimated responses with the tuning doses and measure how the reserved channels behave on factual versus doubled-dose inputs. A useful channel should change strongly under dose edits without substantially perturbing factual states.

```python
print("=== RECORD-IDENTIFIED SIX-LAG ROLLOUT TARGETS ===")
kernels217 = {
    "insulin": np.array([-2.07283356, -2.23466817, -2.14515517,
                        -1.43875875, -1.28052947, -0.90473374]),
    "norepinephrine": np.array([2.95159917, 2.49611187, 1.62799136,
                               1.02869814, 0.34279884, 0.21392442])
}
roll217 = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
dose217 = tune[tune.event.isin(kernels217)].copy()
dose217["start_hour"] = np.floor(dose217.hour + 1e-5).astype(int)
rate_lookup217 = dose217.groupby(
    ["stay_id", "event", "start_hour"]).rate.sum().to_dict()
for edit217, drug217, factor217 in [
        ("insulin_x2", "insulin", 1.0),
        ("insulin_half", "insulin", -.5),
        ("ne_x2", "norepinephrine", 1.0),
        ("ne_half", "norepinephrine", -.5)]:
    selected217 = roll217[roll217.edit.eq(edit217) & roll217.reached]
    shifts217 = []
    for row217 in selected217.itertuples():
        hour217 = int(row217.hour)
        shifts217.append(factor217 * sum(
            kernels217[drug217][lag217 - 1] *
            rate_lookup217.get((row217.stay_id, drug217,
                               hour217 - lag217), 0.0)
            for lag217 in range(1, 7) if hour217 - lag217 >= 12))
    print(edit217, "estimated mean", round(float(np.mean(shifts217)), 4),
          "stay-hours", len(shifts217))
for drug217 in kernels217:
    r217 = train216.loc[train216.event.eq(drug217), "rate"]
    print(drug217, "rate quantiles",
          r217.quantile([0, .05, .5, .95, 1]).round(4).to_dict())

print("\n=== RESERVED CHANNEL FACTUAL AND DOUBLED-DOSE BEHAVIOR ===")
zfact217, tfact217 = posterior_token154(
    zero_weights154, context151["u"])
zdouble217, tdouble217 = posterior_token154(
    zero_weights154, ucf151)
zf217 = zfact217[:, 30:32].numpy()
zd217 = (zdouble217[:, 30:32] - zfact217[:, 30:32]).numpy()
for d217 in [0, 1]:
    mask217 = np.asarray(drug151) == d217
    print("drug", d217, "n", int(mask217.sum()),
          "factual mean", zf217[mask217].mean(0).round(6).tolist(),
          "factual sd", zf217[mask217].std(0).round(6).tolist(),
          "factual quantiles", np.quantile(
              zf217[mask217, d217], [0, .05, .5, .95, 1]).round(6).tolist(),
          "double delta mean", zd217[mask217].mean(0).round(6).tolist(),
          "double delta quantiles", np.quantile(
              zd217[mask217, d217], [0, .05, .5, .95, 1]).round(6).tolist())
for label217, direction217 in [
        ("current", torch.as_tensor(load("z_in.weight"))[:, 30:32].numpy()),
        ("paired_small", dall158),
        ("paired_large", D_best158)]:
    factual_norm217 = np.linalg.norm(zf217 @ direction217.T, axis=1)
    change_norm217 = np.linalg.norm(zd217 @ direction217.T, axis=1)
    print(label217,
          "direction norms", np.linalg.norm(direction217, axis=0).round(3),
          "factual token norm q50/q95/max",
          np.quantile(factual_norm217, [.5, .95, 1]).round(4),
          "dose-change token norm q50/q95/max",
          np.quantile(change_norm217, [.5, .95, 1]).round(4))

print("\n=== CHANNEL-CONSTRUCTION NOTES ===")
lines217 = Path("/work/notes.md").read_text().splitlines()
hits217 = [line217 for line217 in lines217
           if any(key217 in line217.lower() for key217 in
                  ["reserved", "physical-dose", "dose channel", "guarded",
                   "compression", "centered", "centred"])]
print("\n".join(hits217[:22])[:2600])

with open("/work/notes.md", "a") as f217:
    f217.write("\n## Run 217: quantify cumulative targets and channel selectivity\n"
               "- Computing six-lag record-identified rollout targets on the official tuning sample.\n"
               "- Measuring factual versus doubled-dose channel activity and injection sizes for retained candidates.\n"
               "- No weights changed. Python runs: 217; forward passes: 14.\n")
```