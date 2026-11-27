The injection points the right way, but it is not what raises the NLL.
- At readings right after drug bundles, the original model's prediction error regresses positively on my injected shift: slope 1.29 for glucose_finger, 1.27 for map_arterial.
- The error tracks the decoded residuals with about -1.6 per u/h of insulin (glucose) and +2.2 per mcg/kg/min of norepinephrine (MAP). These are close to the g-estimation lag-1 effects.
- So the record really contains a dose signal the model ignores, and my shift is, if anything, slightly too small.
- Repurposing the three units costs nothing (NLL diff 0.0).

The NLL rise therefore comes entirely from how the injection reshapes the predicted distribution, not from its mean. One likely culprit: the averaged direction v̄ also sharpens or widens the prediction.

Next, at those readings, I compare predicted spread and the mass in the observed bin between the original and edited models, and relate the spread change to x. The shape hypothesis is confirmed if the predicted sd changes systematically, e.g. it narrows when glucose belief moves. If the spread is unchanged, I look at the z-prior draws instead.

```python
ia_m = ia
c0 = cen[IDX[ia_m]]; P0 = p0[ia_m, 1:] / p0[ia_m, 1:].sum(1, keepdim=True); P1 = p1[ia_m, 1:] / p1[ia_m, 1:].sum(1, keepdim=True)
mu0 = (P0*c0).sum(1); mu1 = (P1*c0).sum(1)
sd0 = ((P0*(c0-mu0[:,None])**2).sum(1)).sqrt(); sd1 = ((P1*(c0-mu1[:,None])**2).sum(1)).sqrt()
H0 = -(P0*P0.clamp_min(1e-12).log()).sum(1); H1 = -(P1*P1.clamp_min(1e-12).log()).sum(1)
pz0 = p0[ia_m, 0]; pz1 = p1[ia_m, 0]
for c in ('glucose_finger', 'map_arterial'):
    m = torch.from_numpy(codes_[ia_m] == c)
    dsd = (sd1-sd0)[m].numpy(); dH = (H1-H0)[m].numpy(); xx = XA[m.numpy()][:, :2]
    A = np.c_[xx, np.abs(xx), np.ones(len(xx))]
    print(c, 'sd0 mean', sd0[m].mean().item().__round__(3), 'dsd mean', dsd.mean().round(4), 'sd', dsd.std().round(4), '| dsd ~ xI,xN,|xI|,|xN|,1:', np.linalg.lstsq(A, dsd, rcond=None)[0].round(3),
          '| dEntropy mean', dH.mean().round(5), '| p(bin0) diff mean', float((pz1-pz0)[m].mean()))
# NLL change binned by injected shift size
for c in ('glucose_finger', 'map_arterial'):
    m = codes_[ia_m] == c; dd = np.abs(dm[ia_m][m]); dn = (n1-n0)[ia_m][m]
    qs_ = np.quantile(dd, [0, 0.25, 0.5, 0.75, 1.0])
    print(c, '|shift| quartiles', qs_.round(2), 'dNLL by quartile', [round(dn[(dd>=qs_[i])&(dd<=qs_[i+1])].mean(),5) for i in range(4)])
```