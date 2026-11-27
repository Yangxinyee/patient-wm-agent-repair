`tf_eval` confirms it: ablating dims [4, 2, 67] costs +0.0093 on sel2 but nothing on the random 300-stay sets. So sel2 is systematically different. I most likely picked those stays for heavy drug exposure, and these dims seem to hold information that matters mainly for patients on drugs. They looked free only because they were chosen on mostly drug-free stays. The judged factual NLL is on unseen stays that may well include many drug-exposed ones, so I need dims that are cheap on both kinds of stay. This run does two things:
- checks how drug-heavy sel2 is compared with the random sets and the full tuning set;
- re-scans the ablation cost of every one of the 192 dims on both sel2 and dsE.

If several dims cost ≤0.0005 on both sets, I'll rebuild the memory on those. If no dim is cheap on sel2, the reserved-dim design has to change.

```python
print('tr cols', list(tr.columns)[:9], '| tu cols', list(tu.columns)[:10])
print('ds keys', list(ds_s2.keys())[:20])
a_s2=tf_eval(PO,ds_s2,False); a_s2=a_s2[0] if isinstance(a_s2,(tuple,list)) else a_s2
a_E=tf_eval(PO,dsE,False); a_E=a_E[0] if isinstance(a_E,(tuple,list)) else a_E
def dcost(Pd,dsv,a0):
    b=tf_eval(Pd,dsv,False); b=b[0] if isinstance(b,(tuple,list)) else b; return float((b-a0).mean())
t=time.time(); C2=np.zeros(192); CE=np.zeros(192)
for j in range(192):
    Pj=kill(PO,[j],mem=[j]); C2[j]=dcost(Pj,ds_s2,a_s2); CE[j]=dcost(Pj,dsE,a_E)
np.save('/work/dimcost2.npy', np.stack([C2,CE]))
o=np.argsort(np.maximum(C2,CE)); print('time',round(time.time()-t),'s')
print('best by max(sel2,dsE):', [(int(j), round(C2[j],5), round(CE[j],5)) for j in o[:12]])
print('dims 4,2,67:', [(j, round(C2[j],5), round(CE[j],5)) for j in [4,2,67]])
```