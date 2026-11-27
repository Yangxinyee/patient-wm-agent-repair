The register layout is solved: column 0 holds has_last, column 3 holds log(1 + age in seconds)/10, and column 6 holds z_last_value. With that layout my heads match the reported means to 2.1 mg/dL RMSE for glucose and 0.89 mmHg for MAP. That is the size of the 4-draw z noise, so the value head is effectively reproduced exactly.

Next is the core step, which I need in order to see how doubled doses move the state. First I must know how atoms and bundles are built. My guess is that embedding rows with norm 0 are padding, rows near 0.277 (the size of the initial random values) were never trained and so never occur, and the remaining rows are the ones in use. I will print every embedding table's row norms. I will also rebuild the bundles of the sel2 stays from the records, with END atoms at infusion end times, and compare them with the states file's bundle hours, atom counts and drug flags. If the bundle times and atom counts match, the event-to-bundle logic is right. Any mismatch will show where extra atoms come from, for example a discharge atom.

```python
for n in [n for n in NAMES if n.startswith('embed.') and n.endswith('.weight') and n != 'embed.numeric.weight' and 'bins' not in n]:
    print(n, P[n].norm(dim=1).numpy().round(3))
for n in ['embed.bins.value.weight', 'embed.bins.dose.weight', 'embed.bins.rate.weight']:
    print(n, P[n].norm(dim=1).numpy().round(2))
print('numeric col norms', P['embed.numeric.weight'].norm(dim=0).numpy().round(3), 'bias', P['embed.numeric.bias'].norm().item())
# bundles in states for first stay of sel2
sids_S = S['stay_id']; hrs_S = S['hour']; nat_S = S['n_atoms']
s0_ = sids_S[0]
m0 = sids_S == s0_
print('states bundles for', s0_, ':', list(zip(hrs_S[m0].round(4)[:30], nat_S[m0][:30], S['has_insulin'][m0][:30].astype(int), S['has_norepinephrine'][m0][:30].astype(int))))
print(tr[tr.stay_id == s0_].head(40).to_string())
print('last bundles', list(zip(hrs_S[m0].round(4)[-6:], nat_S[m0][-6:])))
```