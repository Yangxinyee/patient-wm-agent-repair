The regression setup completed, but scoring stopped because the cached original losses are tensors rather than NumPy arrays. No weights changed, and there are no candidate results yet. I’ll fix that conversion and finish the held-out screening, then check that the best scalar correction is faithfully represented by the reserved neurons. Agreement would validate the implementation, not yet the rollout effect.

```python
orig_train209 = np.asarray(wm_train145["nll"], dtype=np.float64)
orig_tune209 = np.asarray(wm_tune145["nll"], dtype=np.float64)
map208_candidates = {}
rows208 = []

for ridge208 in (0.003, 0.03):
    for fw208 in (0.1, 0.3, 1.0, 3.0, 10.0):
        for hw208 in (0.0, 0.25, 1.0):
            mat208 = k2 + hw208*kh + 0.25*ki + fw208*q208
            mat208 += ridge208*np.trace(q208)/193*np.eye(193)
            cs208 = la208.solve(mat208, b2+hw208*bh, assume_a="pos")
            add208 = raw_coef208(cs208)
            key208 = (fw208, hw208, ridge208)
            map208_candidates[key208] = map192_cal + add208
            nn208, mm208 = scalar_eval208(
                wm_train145, bt208, add208, mapmask208, cent208)
            tn208, _ = scalar_eval208(
                wm_tune145, bu208, add208, tunemap208, tcent208)
            row208 = dict(
                fw=fw208, half_weight=hw208, ridge=ridge208,
                cv_dNLL=float((nn208-orig_train209)[cvall208].mean()),
                tune_dNLL=float((tn208-orig_tune209).mean()),
                norm=float(np.linalg.norm(map192_cal+add208)))
            for f208 in (0.5, 2.0):
                mc208 = wm_cf146[("norepinephrine", f208)]
                _, cm208 = scalar_eval208(
                    mc208, bc208[("norepinephrine", f208)],
                    add208, mapmask208, cent208)
                for ev208, tag208 in (("map_arterial", "A"), ("map_cuff", "C")):
                    ss208 = cvall208 & (head_meta.event.to_numpy()==ev208)
                    ss208 &= np.asarray(mc208["active"], dtype=bool)
                    dr208 = np.asarray(mc208["delta_rate"])[ss208]
                    dy208 = (cm208-mm208)[ss208]
                    row208[f"{tag208}{f208}_slope"] = float(
                        np.dot(dr208, dy208)/max(np.dot(dr208, dr208), 1e-12))
            rows208.append(row208)

map208_table = pd.DataFrame(rows208)
map208_table.to_csv("/work/map_double_response_frontier.csv", index=False)
print("=== DOUBLING-FOCUSED FRONTIER: LOCAL, NOT ROLLOUT ===")
print(map208_table.query("cv_dNLL < .005 and tune_dNLL < .0047")
      .sort_values("A2.0_slope", ascending=False).head(15).to_string(index=False))
print("\nFull ranges:")
print(map208_table[["cv_dNLL", "tune_dNLL", "A2.0_slope",
                    "A0.5_slope"]].agg(["min", "max"]).to_string())

safe209 = map208_table.query("cv_dNLL < .0042 and tune_dNLL < .0042")
if len(safe209):
    r209 = safe209.sort_values("A2.0_slope", ascending=False).iloc[0]
    bestkey209 = (float(r209.fw), float(r209.half_weight), float(r209.ridge))
    bestcoef209 = map208_candidates[bestkey209]
    full209 = head177_eval(wm_train145, head177_make(bestcoef209))
    sn209, sm209 = scalar_eval208(
        wm_train145, bt208, bestcoef209-map192_cal, mapmask208, cent208)
    print("\nSelected local candidate:", bestkey209)
    print("Scalar/full-head max differences:",
          "NLL", float(np.max(np.abs(full209["nll"]-sn209))),
          "mean", float(np.max(np.abs(full209["mean"]-sm209))))
else:
    print("\nNo candidate meets the conservative local factual screen.")

with open("/work/notes.md", "a") as n209:
    n209.write("\n## Runs 208–209: doubling-focused MAP regression\n"
               "- Fixed a NumPy/tensor loss-conversion error; no weights changed.\n"
               "- Closed-form ridge regressions use fixed head sensitivities, "
               "factual-distribution penalties and cross-drug preservation.\n"
               "- Frontier saved to map_double_response_frontier.csv; "
               "60 fitting and 40 held-out training stays; 21 forward passes used.\n")
```
