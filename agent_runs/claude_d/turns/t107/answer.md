Run 95 corrects my earlier conclusion. With standardized features, the dose surprise is linearly readable right after its bundle from `out` (R² 0.50), from h (0.59) and from the z posterior (0.38). It weakens to 0.19 and 0.30 one bundle later and is gone two bundles later. So the head does see the lag-1 surprise and just weights it too weakly, and the state keeps no longer-term memory of it.

That makes an edit at the head level feasible: add a centred, rescaled readout of the surprise to the predicted mean. Longer lags would then come from the model's own persistence of glucose. Before building it I'll save these findings to the notebook. Then I'll check `vocab.json` for the config, the type indices used by the heads, and the bin centres.

```python
with open('/work/notes.md','a') as f:
    f.write("""
- Run 85-95: TIME ALIGN BUG fixed (float32 hours: use minutes). eps_ins (rate-1.04*base) linearly decodable (standardized ridge a=3):
  state right after ins bundle: out .50, h .59, z .38; next bundle: out .19, h .30; 2 bundles later ~.03. -> dose surprise IS in state at lag1, fades in 2 bundles (no long memory).
  Earlier 'z 0.00/out 0.12' R2s were ridge over-shrinkage artefacts (unscaled features).
- Factual baseline (fwd4, selS train 100): NLL 1.549; resid sd gf 15.0, gl 14.6, map_a 5.2, map_c 7.7.
- Plan: head-level centred readout correction (value head) + rely on model glucose persistence for later lags; tune scale with dose_edit 'both'.
""")
V=json.load(open('/model/vocab.json'))
print(list(V.keys()))
print(V.get('config'))
```