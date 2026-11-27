The dose-dependent part of a drug atom's embedding is large: within-code RMS 3.4 for insulin and 2.8 for norepinephrine, against code-embedding norms of about 2.4. Yet the model's dose response is tiny, so downstream layers have learned to largely ignore it for readings. The categorical tables have a zero row 0 (absent), and rows with norm about 0.27 look unused.

This suggests a clean way to remove the dose pathway without code-specific compensation. The median record of each drug (insulin 2 u/h; norepinephrine 0.53 mcg/kg/min, dose 2.544 mg) has z = 0 and bin 8 for both dose and rate. So I can:
- zero the z_dose and z_rate columns of `embed.numeric`, which only drug atoms use;
- set rows 1–16 of `embed.bins.dose` and `embed.bins.rate` to their row 8.

Then every drug record is embedded exactly like a median-dose record of its own drug: an input the network has seen many times. Dose edits then change nothing, so all four readouts become exactly 0. Readings' registers use bin 0, which stays unchanged. The open question is factual NLL, which I will test with the tool.

Next I will build and save this edit and confirm, on the training records, that the factual and doubled/halved embeddings now coincide.

```python
t = load('embed.numeric.weight'); print(type(t), t.dtype if hasattr(t,'dtype') else None)
import torch
Wn = LO('embed.numeric.weight').copy(); Wn[:,2] = 0.0; Wn[:,4] = 0.0
Bd2 = LO('embed.bins.dose.weight').copy(); Bd2[1:,:] = Bd2[8,:]
Br2 = LO('embed.bins.rate.weight').copy(); Br2[1:,:] = Br2[8,:]
conv = (lambda a: torch.tensor(a, dtype=torch.float32)) if isinstance(t, torch.Tensor) else (lambda a: a.astype(np.float32))
save_edit('embed.numeric.weight', conv(Wn)); save_edit('embed.bins.dose.weight', conv(Bd2)); save_edit('embed.bins.rate.weight', conv(Br2))
print(edited())
# verify invariance of drug-atom embedding term under x2 / x0.5 for all observed rates
for code in ['insulin','norepinephrine']:
    grp = stats[code]['grp']
    for f in [1.0, 2.0, 0.5]:
        zr, br = zb(grp.rate.values*f, code, 'rate'); zd, bd = zb(grp.dose.values*f, code, 'dose')
        Dn = L('embed.numeric.weight')[:,2][None,:]*zd[:,None] + L('embed.numeric.weight')[:,4][None,:]*zr[:,None] + L('embed.bins.dose.weight')[bd] + L('embed.bins.rate.weight')[br]
        print(code, f, "max dev from median-atom term:", np.abs(Dn - (Bd2[8]+Br2[8])[None,:]).max())
```