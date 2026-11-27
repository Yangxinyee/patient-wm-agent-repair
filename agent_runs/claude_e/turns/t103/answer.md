Forward pass 6 returned states for 100 train stays. I made an error, though: I typed the stay list by hand instead of copying batch 1 from /work/sel_tr2.json. So these are 100 arbitrary train stays, not my drug-rich batch, and one stay overlaps batch 0.

They are still valid states from the original model and can be used for fitting. They just carry less dose variation.

To avoid this in future, I'll print the saved batch lists and copy them exactly. I'll also record which stays pass 6 actually covered.

```python
S6 = np.load('/work/fwd_6/states_original.npz', allow_pickle=True)
st6 = sorted(set(int(s) for s in S6['stay_id']))
print('fwd_6 stays:', len(st6), 'insulin bundles', int(np.sum(S6['has_insulin'])), 'NE bundles', int(np.sum(S6['has_norepinephrine'])))
json.dump({'fwd_5': batches[0], 'fwd_6': st6}, open('/work/states_stays.json','w'))
with open('/work/notes.md','a') as fh:
    fh.write("""
- Runs 93-95: batch0 (fwd_5, 100 train stays both drugs): model dev coefs [-0.39 .24 .08 .11 .22 .34] (se .10) vs truth [-2.04 -1.23 -.25 .38 .80 .69].
  dev1 decodable from head q only CV R2 ~0.22-0.27 (out_prev .28, full state .45); dev2 ~0.05. Model's -0.39 ~= true*R2 -> representational bottleneck.
  Plan A: lag-1 unit-gain readout correction in value head (2 GELU units per drug, code-gated), coef ~ -2.25 (to match total sum of one-step coefs), cost ~0.003.
- fwd_6: by mistake states for 100 hand-typed train stays (not batch1); saved in /work/states_stays.json. Use batches from /work/sel_tr2.json for next requests.
""")
print(json.dumps(batches[1]))
```