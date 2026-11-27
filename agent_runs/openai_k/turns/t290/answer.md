The glucose-only edit improves insulin rollouts to −10.20/+4.44 while passing factual loss, but norepinephrine remains unrepaired. I’ll now fit dose features that distinguish changes in the mean across recurrent slots from changes that cancel between slots. This may provide stronger dose sensitivity with less factual disruption. The hypothesis is supported only if the new features improve multiplier scaling and survive official factual validation.

```python
r257_F = np.asarray(r243_E, dtype=np.float64)
r257_Fm = r257_F.mean(axis=1)
r257_Fw = (r257_F-r257_Fm[:, None, :]).reshape(-1, 192)
r257_Cm = r257_Fm.T @ r257_Fm / len(r257_Fm)
r257_Cw = r257_Fw.T @ r257_Fw / len(r257_Fw)
r257_Dm = r245_D.reshape(-1, 4, 192).mean(axis=1).astype(np.float64)
r257_Ym = r245_Y.reshape(-1, 4, 2).mean(axis=1).astype(np.float64)
r257_weights = np.zeros(len(r257_Dm))
for i, key in enumerate(r245_keys):
    block = slice(i*320, (i+1)*320)
    contrast_rms = np.sqrt(np.mean(np.sum(r257_Ym[block]**2, axis=1)))
    r257_weights[block] = 1.0/max(contrast_rms, 0.02)**2
r257_weights /= r257_weights.mean()
r257_Cd = r257_Dm.T @ (r257_weights[:, None]*r257_Dm)/len(r257_Dm)
r257_rhs = r257_Dm.T @ (r257_weights[:, None]*r257_Ym)/len(r257_Dm)
r257_ridge = 1e-6*np.trace(r257_Cd+r257_Cm)/192
r257_fits = {}

print("CURRENT feature factual slot RMS:",
      np.sqrt(np.mean((r257_F.reshape(-1,192)@r245_V)**2,axis=0)).round(5))
print("CURRENT feature factual mean RMS:",
      np.sqrt(np.mean((r257_Fm@r245_V)**2,axis=0)).round(5))
print("CURRENT V matches penalties:",
      {k: round(float(np.linalg.norm(v-r245_V)),6) for k,v in r245_fits.items()})

print("\n=== SLOT-MEAN FEATURE FITS ===")
for within in [0.001, 0.01, 0.1, 1.0]:
    for penalty in [0.1, 1.0, 10.0]:
        cov = r257_Cd + penalty*(r257_Cm+within*r257_Cw)
        vv = np.linalg.solve(cov+r257_ridge*np.eye(192), r257_rhs)
        r257_fits[(within, penalty)] = vv
        means = np.array([
            (r257_Dm[i*320:(i+1)*320]@vv).mean(axis=0)
            for i in range(10)
        ])
        fm = np.sqrt(np.mean((r257_Fm@vv)**2, axis=0))
        fs = np.sqrt(np.mean((r257_F.reshape(-1,192)@vv)**2, axis=0))
        ratios = []
        for i,key in enumerate(r245_keys):
            col = 0 if key[0]=="insulin" else 1
            wanted = r257_Ym[i*320:(i+1)*320,col].mean()
            ratios.append(float(means[i,col]/wanted))
        print("within",within,"penalty",penalty,
              "fact mean RMS",fm.round(4).tolist(),
              "slot RMS",fs.round(4).tolist(),
              "dose slopes [I2,I0,Ihalf,I1.5,I3,N2,N0,Nhalf,N1.5,N3]",
              np.round(ratios,3).tolist())

r257_factual_delta = (
    pd.read_csv("/work/fwd_20/factual_edited.csv")["nll"].mean()
    -pd.read_csv("/work/fwd_20/factual_original.csv")["nll"].mean()
)
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 257: slot-mean factual constraints\n"
        f"- Glucose-only candidate passes forward 20 factual, delta {r257_factual_delta:.8f}; "
        "forward 21 shifts insulin -10.203/+4.439, NE +0.327/-0.048.\n"
        "- Fitting multiplier-normalized physical-rate contrasts with separate penalties on "
        "slot-mean and within-slot factual energy. Hypothesis: avoid penalizing cancellation "
        "directions that have little readout effect.\n"
        "- No weights changed. Python runs: 257; forward passes: 21.\n"
    )
```
