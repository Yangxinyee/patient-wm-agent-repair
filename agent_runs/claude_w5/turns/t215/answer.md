The transparent step matches cstep_h exactly (zero error on all 52 bundles). Its sixth output is h_new. I now have full visibility into the prelude (attention probabilities, the slot residual before and after the FFN, the FFN units) and hooks to add to P or h_new.

Next I record what the drug atoms look like and collect teacher-forced steps for the 100 training stays in bundles_B. At the drug bundles I measure:
- how much the scale factors that LayerNorm applies (std of the slot residual before the FFN and of P) vary across bundles;
- how large the attention output is relative to the state.

If those scales vary only a few percent, a prelude-FFN unit can compute "rate minus policy mean" with a fixed linear readout and write a fixed direction. If they vary a lot, I need a scale-robust design.

```python
from collections import Counter
sids_B = list(bundles_B.keys())
steps_all = {}
for s in sids_B:
    steps_all[s] = run_tf(bundles_B[s], keep=True)
# empty-action u: most common u vector
uc = Counter()
for s in sids_B:
    for d in steps_all[s]: uc[tuple(np.round(d['u'].numpy()[:6], 8))] += 1
key0, n0 = uc.most_common(1)[0]
for s in sids_B:
    for d in steps_all[s]:
        if tuple(np.round(d['u'].numpy()[:6], 8)) == key0: u_empty = d['u'].clone(); break
    else: continue
    break
nb = sum(len(v) for v in steps_all.values())
print('bundles', nb, 'empty-u count', n0)
# find a drug bundle and print its atoms
done = False
for s in sids_B:
    for k, d in enumerate(steps_all[s]):
        if (d['u'] - u_empty).norm() > 1e-6:
            ats = bundles_B[s][k][1]
            print('drug bundle sid', s, 'k', k, 't', bundles_B[s][k][0], 'n atoms', len(ats))
            for a in ats[:3]: print('  ', {kk: (str(vv)[:40]) for kk, vv in (a.items() if isinstance(a, dict) else enumerate(a))})
            done = True; break
    if done: break
# scale stats at drug bundles
S1, SP, AO, HF = [], [], [], []
for s in sids_B[:60]:
    for d in steps_all[s]:
        if (d['u'] - u_empty).norm() > 1e-6:
            R = {}; mystep(d, R=R)
            x1s = R['x1'][:4]
            S1.append(x1s.std(1).numpy()); SP.append(R['P'].std(1).numpy())
            ao_ = (R['x1'][:4] - R['toks'][:4]); AO.append(ao_.norm(dim=1).numpy()); HF.append(R['hf'].norm(dim=1).numpy())
S1, SP, AO, HF = map(np.array, (S1, SP, AO, HF))
print('drug bundles', len(S1))
print('std(x1) slots mean', S1.mean(0).round(2), 'CV', (S1.std(0)/S1.mean(0)).round(3))
print('std(P)  slots mean', SP.mean(0).round(2), 'CV', (SP.std(0)/SP.mean(0)).round(3))
print('|attn out| mean', AO.mean(0).round(1), '|hf| mean', HF.mean(0).round(1))
```