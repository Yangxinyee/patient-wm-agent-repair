This decides what is possible. In the summary vector `out` just before each reading:

- **Insulin:** the doubled dose is almost invisible at glucose readings. The GLS signal-to-noise ratio is 0.11 per unit², so the noise is about 3 insulin units against a typical dose change of 1.7. Reading bundles come right after the drug bundles (hours 0, 0.083, 1, 1.083…), so this is the state immediately after the drug bundle. The model's action-token representation of insulin is saturated (run 279), yet its own step barely carries it into the state. No readout or encoder edit can create that signal. Fixing it would need a full rebuild of the prelude attention, and 16 runs is not enough for that.
- **Norepinephrine:** the dose is strongly present in `out` (SNR 14.8 per unit², noise 0.26 against a typical change of about 0.5). The model simply barely uses it for MAP. The value-head readout I planned would need the exact register layout (`last_numeric` is 192×8 and was fitted empirically before the crash), so it is out of reach now.

The cheap lever left is to make the existing norepinephrine signal larger at its source. I will expand the rate and dose part of every drug atom's embedding around the median bin, which happens to be bin 8 for both drugs and both attributes. That means scaling the z_rate and z_dose columns of `embed.numeric` and expanding the rate/dose bin rows away from row 8, by a factor k. The reading tables and row 0 are untouched, so reading atoms are unaffected.

Insulin's representation is already saturated, so this should mainly strengthen norepinephrine. I will test k = 6 with a dose-edit pass on the same 100 tuning stays as forward pass 11, then a factual pass.

```python
import numpy as np, torch, pandas as pd
EN = TT(load_original('embed.numeric.weight')).astype(np.float64)
BR = TT(load_original('embed.bins.rate.weight')).astype(np.float64)
BD = TT(load_original('embed.bins.dose.weight')).astype(np.float64)
print('numeric col norms', np.linalg.norm(EN, axis=0).round(3), '| rate bin dev norms', np.linalg.norm(BR[1:] - BR[8], axis=1).round(2))
def set_inamp(k):
    en = EN.copy(); en[:, [2, 4]] *= k
    br = BR.copy(); br[1:] = BR[8] + k * (BR[1:] - BR[8])
    bd = BD.copy(); bd[1:] = BD[8] + k * (BD[1:] - BD[8])
    save_edit('embed.numeric.weight', torch.tensor(en, dtype=torch.float32))
    save_edit('embed.bins.rate.weight', torch.tensor(br, dtype=torch.float32))
    save_edit('embed.bins.dose.weight', torch.tensor(bd, dtype=torch.float32))
set_inamp(6.0); print('edited:', edited())
de = pd.read_csv('/work/fwd_11/dose_edit_original.csv'); tids = sorted(int(s) for s in de.stay_id.unique()); print(len(tids)); print(tids)
with open('/work/notes.md', 'a') as f:
    f.write('- run284 (out-space GLS from fwd12 emulation): INSULIN dose barely in out at glucose readings: SNR .11/unit^2 (noise 3 units) -> insulin readout NOT viable; info lost in drug-bundle step (prelude). NE strongly in out at MAP readings: SNR 14.8/unit^2 (noise .26) but value head ignores it.\n  Readings bundles at integer hours, drug bundles at h+.083, so the next reading follows the drug bundle directly.\n- run285: INPUT-LEVEL dose amplification (shared): embed.numeric cols 2,4 (z_dose,z_rate) x k, bins.rate/dose rows 1..16 expanded around row 8 (median bin for both drugs) by k. set_inamp(k). Testing k=6 (dose_edit tuning + factual).\n')
```