The honest out-of-sample cost is worse than I thought. `RR` holds 5214 rows from the sel1 tuning stays (src '1', matching `X`) plus 5139 rows from a train set A whose states are elsewhere. That mix caused the fake blow-up.

On the sel1 rows alone, the saved edit costs **+0.0080 per reading** against the original, which is over the 0.005 budget:
- glucose_finger: +0.0056 per row;
- map_arterial: +0.0216 per row;
- lab glucose: −0.0008 per row.

On the random f3 set the tool measured only +0.0034. sel1 was probably chosen for drug-heavy stays, so the cost depends strongly on how many readings follow a drug. The NE part is the expensive one, and it only helps NE ×½.

Before going further I'll record all of this in the notebook, since the conversation may restart. My next step after that is to split the sel1 cost by insulin part and NE part, per row type. Then I'll reweight to a random-stay mix and choose gains under the budget.

```python
notes_add_txt = r"""

## Runs 176-188 (after restart 5)
- BUGFIX: global `lin` had been overwritten by a 1-arg lambda (kept as lin_r); restored lin(x,name)=x@W.T+b. eval_edit/head_run work again.
- gated3(a_ins,a_ne) returns dict {'W0','b0','W2','TR'} (short keys; KMAP maps to tensor names). Saved edit == gated3(-1.9,3) (hash 262704e88f).
  It repurposes 4 head units; gated3(0,0) != original (costs ~0.0004-0.0013 by itself).
- fwd_10: dose_edit both, tuning f3 (2020000000..99), edit gated3(-1.9,3). orig: ins x2 -1.87, half +.76, NE x2 -.51, half -.70.
  edited: -9.85, +4.13, -0.80, -1.91. MY TRUTH on same rows (truth_ins/truth_ne): -13.38, +6.69, +4.89, -2.45.
  Profile by k=hours since first reached: truth -3.0,-6.8,-9.6,-11.9,-13.6,-14.9,-16.2,-16.4,-17.3,-17.5,-18.3,-20.3; edited -3.35,-4.6,-5.7,-7.5,-8.1,-9.0,-10.6,-11.2,-13.4,-14.5,-16.7,-18.9.
  -> edited ramps ~linearly, truth front-loaded. Only the mean is judged. Linear scaling suggests a_ins ~ -2.7 for x2.
  NE x2 edit fails (rates >1 are OOD bins 13-16; bins tables embed.bins.rate etc are SHARED across codes, insulin uses bins 13-16 -> no free fix).
- fwd_11: factual both on f3: orig 1.63461 -> edited 1.63798 (+0.0034). per-row: gf +.0062 (n1538), gl +.0013 (488), ma +.0053 (1109), mc 0.
- RR = 5214 sel1 rows (src '1', states X=fwd_2) + 5139 train-A rows (src 'A', states NOT in X!). Use RR[src=='1'] with X only.
- Offline head_run on sel1 rows validated vs tool on 38 overlapping stays (corr dpm .975-.986, dn similar).
- Offline OUT-OF-SAMPLE (sel1) cost of saved edit vs orig: +0.0080/reading (gf +.0056/row, ma +.0216/row, gl -.0008). sel1 is drug-heavy;
  random f3 gave +.0034 in tool. eval_edit (train datasets, in-sample readouts) is optimistic. Factual cost depends on drug-row share -> risky.
- NEXT: decompose cost (insulin vs NE part) per row type on sel1; reweight to random-stay composition; pick a_ins/a_ne; check with tool on all 400 tuning stays.
"""
with open('/work/notes.md','a') as fh: fh.write(notes_add_txt)
z_ = np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
print(list(z_.keys()), {k: z_[k].shape for k in z_.keys()})
```