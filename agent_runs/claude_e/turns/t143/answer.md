The gated edit now behaves as designed and fits inside the budget:

- **Pairs.** The insulin pair equals its readout on glucose rows after insulin and is exactly 0 everywhere else. The norepinephrine pair equals its readout on arterial rows and is 0 elsewhere. The two non-zero errors in the table are artefacts of my group labels, which mix on and off rows.
- **Pre-activations.** The largest is 277, which is safe in float32.
- **Factual cost.** At a_ins = −1.9 and a_ne = +3 the offline cost is 0.0052 nats per reading, against 0.018 for the ungated version.
  - Arterial MAP under norepinephrine accounts for about 0.0035 of it, because the norepinephrine readout is weak (R² 0.19).
  - The insulin side costs about 0.001.
- **Gated rows.** Glucose rows with no insulin and cuff rows are unaffected, apart from the small cost of freeing the four units.
- **One-step slopes.**

  | Prediction | Slope per unit lag-1 deviation |
  |---|---|
  | Finger glucose | −2.17 |
  | Lab glucose | −1.44 |
  | Arterial MAP | +3.27 |

The next step is the test that matters: does a one-step correction of this size give the right rollout effect? I will record these findings in the notebook and save this edit. After that I will run dose_edit with "both" on the sel1 tuning stays.

My truth estimates on sel1 are:

| Edit | Truth estimate |
|---|---|
| Insulin ×2 | about −13.3 to −14.3 |
| Insulin ×½ | about +6.7 to +7.1 |
| Norepinephrine ×2 | about +4.5 |
| Norepinephrine ×½ | about −2.3 |

Results near these would support the approach. Effects far smaller, near the one-step size, would mean the model's own persistence does not carry the correction forward.

```python
notes_add = """
## Runs 105-131 (after restart 4)
- fwd_4=trA_A (sel_train.json A), fwd_5=tr2_0 (=batch0), fwd_6=hand-typed (states_stays.json), fwd_7..9=tr2_1..3. All 'states' of original.
- Kernel: datasets[0..4]=(states, rows) for fwd_5..9; base_out[i]=(pm,nll,P,hidden16) of original head; head_run(Xs,R,Wd) = local value head (16 z-draws, matches tool).
- Tensor names: W0=attr_heads.value_bin.0.weight, b0=attr_heads.value_bin.0.bias, W2=attr_heads.value_bin.2.weight, TR=embed_transition.value_bin.weight. ORIG dict is float64.
- Old ungated make_edit (a_ins=-1.9,a_ne=3) costs 0.018 nats/reading: noise on glucose rows w/o insulin in prev bundle + cuff MAP. Rejected.
- Per-draw head inputs: head_run with W0=I, b0=100 -> hidden=wc+100. WCall(16,27963,192) for datasets 0-4; A_ has code, hi/hn (drug in prev bundle), dev1, nedev1, stay.
- Readout facts: dose info sits in low-variance directions of the head input; target-only ridge readouts explode off-target (+-130). Fix = generalized ridge with off-target penalty (lam 0.01) + code-embedding-invariant weights.
- NEW EDIT gated3(a_ins,a_ne): units [26,41,131,159]. Insulin pair u1-u2 = unit-gain readout of dev1 (ROi5, R2 .52) gated by glucose code gate (gvec,K=40) x insulin-in-prev-bundle gate (Gi3,A=30,M=-10).
  NE pair u3-u4 = readout of nedev1 (ROn4, R2 .19) gated to map_arterial only (Gn2, B=10, M=-1). Logit directions tg/tm. Pair outputs verified exact on all 16 draws; max pre-act 277.
- Offline cost a_ins=-1.9,a_ne=3: 0.0052/reading (ma on-rows .013 -> ~.0035 of total; gf .0029 on rows). Lag-1 slopes: gf -2.17/u/h dev, gl -1.44, ma +3.27 per unit NE-rate dev.
- Next: fwd dose_edit both on sel1 with gated3(-1.9,3) to calibrate a_ins, a_ne against truth est (ins x2 -13.3..-14.3, half +6.7..7.1; NE x2 +4.53, half -2.27).
"""
open('/work/notes.md','a').write(notes_add)
Ecur=gated3(-1.9,3.0)
for k,v in Ecur.items(): save_edit(names_map[k][0], v.float())
print('saved',edited())
for k in Ecur: print(k,'roundtrip maxdiff',float((torch.as_tensor(load(names_map[k][0])).double()-Ecur[k].double()).abs().max()))
```