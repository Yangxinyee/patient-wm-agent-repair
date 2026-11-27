E3-ins behaves well on the GPU:
- **Factual arm.** The hour-13 bias of the edited model's factual arm is +0.8 mean and +0.4 median, against +14 for E2. Over the hours it drifts only to about +3, and that drift brings it closer to the real readings, which sat about 3 mg/dL above the original model's rollout.
- **Linearity.** Both edits respond linearly.

The hourly shape is too strong early (hour 13: −7.0 against −3.2), matches in the middle, and is too strong late (hour 24: −24 against −17). The cause is that the lag-1 channel carries the summed shortfall, while the model then carries shifts forward more persistently than the true effect decays. Only the mean over hours 13–24 is judged, so the practical fix is to scale C down. On a linear rescaling that gives C ≈ −2.7 for x2 and −2.9 for x1/2, about −2.8.

Before re-running insulin, I want to settle whether norepinephrine is reachable at all. Its with_code decoders reach only R² 0.15–0.21. Nearly all of the norepinephrine effect is still missing: the readouts are −0.05 and −0.48 against roughly +4.5 and −2.3.

I will check how well the full 992-number state just before each MAP reading predicts the norepinephrine signals: the rate deviation, the one-step surprise, and the rate itself. I will check `out`, the slot mean and the individual slots separately.
- If the full state predicts them well (R² 0.5 or more) while with_code does not, the information is lost when the view is built. An edit in the view path could then help.
- If even the full state is poor, norepinephrine is out of reach with weight edits of this kind, and I will spend the remaining effort on insulin precision.

```python
with open('/work/notes.md','a') as f:
    f.write("- Fwd10 = dose_edit both sel1 with E3-ins (C=-3.32): ins_x2 -16.55 (tgt -13.83), ins_half +7.86 (tgt +6.92), NE unchanged. Factual-arm bias h13 +0.8 (bf16 fix works). Profile: early & late overshoot, mid ok; model persistence > true decay. Linear calibration -> C ~ -2.8.\n")
pr_rows = rd_.prev_row.values.astype(int)
mkn_ = mkn & (pr_rows >= 0)
Xs_full = XALL[pr_rows[mkn_]]
from sklearn.model_selection import GroupKFold, cross_val_predict
gkf = GroupKFold(5)
def cvr2_blk(X_, y_, al_=1.0):
    Xz = (X_ - X_.mean(0)) / (X_.std(0) + 1e-6)
    p_ = cross_val_predict(Ridge(alpha=al_), Xz, y_, cv=gkf, groups=sidv[mkn_])
    return 1 - np.mean((p_-y_)**2)/y_.var()
ne1 = rd_.ne_1.fillna(0).values
blocks = {'out': slice(0,192), 'h_all': slice(192,960), 'z': slice(960,992), 'full': slice(0,992)}
for tnm, y_ in [('D_ne_1', D1[mkn_]), ('eps_ne_1', n1[mkn_]), ('ne_rate_1', ne1[mkn_])]:
    res__ = {b_: round(cvr2_blk(Xs_full[:, s_], y_, 10.0), 3) for b_, s_ in blocks.items()}
    hm = Xs_full[:, 192:960].reshape(-1, 4, 192).mean(1)
    res__['h_mean'] = round(cvr2_blk(hm, y_, 10.0), 3)
    res__['wc'] = round(cvr2_blk(WC0[mkn_], y_, 10.0), 3)
    print(tnm, res__)
```