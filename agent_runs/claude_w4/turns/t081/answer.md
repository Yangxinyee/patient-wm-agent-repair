The exhaustive search over the constant tables still gives 0.33 at best. So the constant part is not the culprit either. Both the code-dependent and the value-dependent parts must be represented differently than I assumed.

The cleanest remaining hypothesis is that a reading's value fills other numeric or bin slots as well, for example the rate or dose bin tables or the z_dose and z_rate inputs. Another possibility is that the per-code median and IQR differ from what I used.

Next I'll print the glucose normalisation entries and search over:
- which bin tables receive the value bin (value only, or also rate and/or dose);
- which numeric slots receive z;
- the unit rows.

A match near 1e-6 completes e_k. If there is still no match, I'll stop replicating e_k. The edit can then rely only on u_k, which is exact, plus ctx, which is exact and computed from the states.

```python
for c_ in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    d = num[(c_,'value')]; print(c_, d['median'], d['iqr'], d['fine_bin_edges'][:4], '...')
resC = []
binsets = {'V':[VB], 'VR':[VB,BR], 'VD':[VB,BD], 'VRD':[VB,BR,BD], 'R':[BR], 'D':[BD]}
numsets = {'v':lambda z:[z,1,0,0,0,0], 'vd':lambda z:[z,1,z,1,0,0], 'vr':lambda z:[z,1,0,0,z,1], 'vdr':lambda z:[z,1,z,1,z,1], 'r':lambda z:[0,0,0,0,z,1], 'd':lambda z:[0,0,z,1,0,0]}
for bs, ns, ug, um, kqi in itertools.product(binsets, numsets, [0,2,5],[0,2,5],[None,0]):
    A = []
    for _, (c_, v_) in singles:
        d = num[(c_,'value')]; b = fbin(v_, d['fine_bin_edges']); z = zsc(v_, d)
        a = E_['code'][cid[c_]] + E_['kind'][0] + E_['ekind'][1] + sum(T[b] for T in binsets[bs]) + NW @ torch.tensor(numsets[ns](z), dtype=torch.float32) + NB
        a = a + CAT['unit'][ug if c_.startswith('glucose') else um] + (E_['kq'][kqi] if kqi is not None else 0)
        A.append(a)
    resC.append((float((e_from_atoms_batch(torch.stack(A))[:, e_c] - TS).abs().max()), bs, ns, ug, um, kqi))
resC.sort(key=lambda t: t[0]); print(resC[:4])
```