The probes settle it: the head-level route does not work on the tool.

- **Drug 1:** the tool's decoder value on finger readings is −29 ± 7.3 pre-activation units, against 0 ± 0.7 in my replica. The spread is 10× the replica's whole signal range, split between and within stays, and essentially uncorrelated with residuals (−0.07). The replica's head inputs are slightly but systematically wrong exactly along the directions a minimum-variance decoder relies on.
- **Drug 2:** the constant probe produced essentially no shift on arterial readings. The ratio of about 1 in early hours is just the 4-decimal rounding of the output. So this gate does not open on the tool, and v5's small arterial effect was leakage.

With 9 Python runs left I can't rebuild a trustworthy replica of the head inputs. I'll switch to an edit judged entirely by the real model: amplify the model's own dose input. I'll scale the z_dose and z_rate columns of `embed.numeric` by 3. Those columns enter only drug records, so the model sees dose differences three times larger through whatever pathway it already uses.

All head edits are dropped first, and the notebook is updated.

The dose_edit "both" pass then shows whether this moves the four readouts toward the targets: drug-1 effects larger and of the same sign as now, and the drug-2 effects likewise. If it does, a factual pass checks the cost. If the readouts barely move or flip sign, this route is refuted and I keep the original model.

```python
for n in list(edited()): drop_edit(n)
nm=[n for n in NAMES if n.startswith('embed.numeric')]; print([(n,tuple(SHAPES[n])) for n in nm])
Wnum=np.array(load_original('embed.numeric.weight'))
lam=3.0; Wl=Wnum.copy()
if Wl.shape[1]==6: print('col norms',np.round(np.linalg.norm(Wl,axis=0),3)); Wl[:,2]*=lam; Wl[:,4]*=lam
else: print('row norms',np.round(np.linalg.norm(Wl,axis=1),3)); Wl[2]*=lam; Wl[4]*=lam
save_edit('embed.numeric.weight', torch.tensor(Wl,dtype=torch.float32)); print('edited', edited())
open('/work/notes.md','a').write("\n## Session 9 (runs 279-291)\n- fwd_7: v4 BROKEN on tool (NLL 46): decoder term had constant offset ~145 inside LN'd q -> z-draw scale noise flips gates.\n- v5 offset-free z-orthogonal decoders (fwd_8): lab/cuff fixed, art +0.016, finger broken (dmu ~+83).\n- Probes P1/P2 (fwd_9/10, ratio method): tool decoded drug-1 value -29+-7.3 vs replica 0+-0.7 -> head decoding UNUSABLE; drug-2 gate does not open on tool.\n- Run 291: head edits dropped; testing embed.numeric z_dose/z_rate column x3 (model's own dose pathway), judged by tool dose_edit + factual.\n")
```