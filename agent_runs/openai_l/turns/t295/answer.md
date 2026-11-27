The existing insulin decoder varies much more on independent factual states than on its fitting states, suggesting poor generalization. I’ll refit small linear dose decoders using broader factual coverage and paired dose contrasts. The hypothesis is supported if factual variance falls on both samples while held-out dose contrasts remain accurate.

```python
decoder262 = {}
for drug262 in ("insulin", "norepinephrine"):
    mt262 = family204[drug262] & fitmask179
    mv262 = family204[drug262] & valmask179
    mq262 = familyq261 if drug262 == "insulin" else ~familyq261
    xt262 = Xraw204[:, mt262].reshape(-1, 193)
    xq262 = X225[:, mq262].reshape(-1, 193)
    mu262 = 0.5*(xt262.mean(axis=0)+xq262.mean(axis=0))
    ct262, cq262 = xt262-mu262, xq262-mu262
    gf262 = 0.5*(ct262.T@ct262/len(ct262)+cq262.T@cq262/len(cq262))
    gd262 = np.zeros((193, 193))
    rd262 = np.zeros(193)
    count262 = 0
    for key262, p262 in paired214.items():
        if key262[0] != drug262:
            continue
        use262 = p262["own"] & fitmask179[p262["ids"]]
        dx262 = p262["dx"][:, use262].reshape(-1, 193)
        dr262 = np.broadcast_to(p262["dr"][use262], (8, use262.sum())).reshape(-1)
        gd262 += dx262.T@dx262
        rd262 += dx262.T@dr262
        count262 += len(dx262)
    gd262 /= count262
    rd262 /= count262
    pv262 = paired214[(drug262, 2.0)]
    uv262 = pv262["own"] & valmask179[pv262["ids"]]
    dv262 = pv262["dx"][:, uv262]
    tv262 = pv262["dr"][uv262][None, :]
    for lam262 in (0.01, 0.03, 0.1, 0.3, 1.0):
        for ridge262 in (1e-9, 1e-8, 1e-7):
            d262 = np.linalg.solve(gd262+lam262*gf262+ridge262*np.eye(193), rd262)
            d262[-1] -= mu262@d262
            prediction262 = dv262@d262
            slope262 = float(np.mean(prediction262*tv262)/np.mean(tv262**2))
            metrics262 = (
                float((Xraw204[:, mt262]@d262).std()),
                float((Xraw204[:, mv262]@d262).std()),
                float((X225[:, mq262]@d262).std()),
                slope262,
                float(np.sqrt(np.mean((prediction262-tv262)**2))),
                float(np.linalg.norm(d262))
            )
            decoder262[(drug262, lam262, ridge262)] = {"d": d262, "metrics": metrics262}
            print(drug262, lam262, ridge262,
                  "SD fit/val/tune, slope, RMSE, norm:", np.round(metrics262, 4))

with open("/work/notes.md", "a") as f262:
    f262.write("\n## Runs 261–262 / broaden dose-decoder anchoring\n"
               "Rejected antisymmetric-pair readout: independent offline factual delta >.010. "
               "Old insulin decoder factual SD is 1.12 on fitting data but 2.79 on independent states. "
               "Fit small linear decoders analytically with train-fit and independent factual covariance "
               "penalties plus physical paired-dose contrasts. No model weights changed.\n")
```
