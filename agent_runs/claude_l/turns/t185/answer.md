The prelude's attention is effectively hard. At drug bundles, heads 0–2 put weight about 1.0 on the u token, so the slots copy a linear projection of normalised u, while head 3 attends to the slots. At reading bundles, heads 1–2 copy e and one head copies u. The z token gets almost no weight anywhere, which is why z is dead even though `z_in` is not zero. Together with the fast forgetting of h, the realistic first design is a lag-1 dose effect, with the later hours carried by the model's own feedback from its drawn readings. Whether that can reach the true response depends on how the model's rollout propagates a one-hour glucose shift. I'll measure it with the replica: double only a single insulin record (the first one at or after hour 12) and track the glucose_finger shift over hours 13–24. If the propagation decays slowly (flat), a lag-1-only edit gives a persistent response. True dynamics would show a peak at 2–3 h and then recovery.

```python
code_ro2 = code_ro.replace("if cond is not None and t>=t_edit:\n        dcode,f=cond; ua=[edit_atom(a,f) if a['code']==dcode else a for a in ua]",
 "if cond is not None and t>=t_edit:\n        dcode,f=cond[0],cond[1]\n        if len(cond)>2 and not (cond[2]<=t<cond[3]): return set_enc(W0c[0],[fix_atom2(a) for a in ua],'encode_action')\n        ua=[edit_atom(a,f) if a['code']==dcode else a for a in ua]")
exec(code_ro2)
# single-dose edit: only the first insulin record in [12,13)
prop=[]; prop_full=[]
for s in b1[:60]:
    B_=BUN[s]
    if not any(12<=B_[k][0]<13 and any(a['code']==8 and a['kind']==0 for a in B_[k][1]) for k in range(len(B_))): continue
    ua=[a for k in range(len(B_)) if 12<=B_[k][0]<13 for a in B_[k][1] if a['code']==8 and a['kind']==0]
    d0=ua[0]['num'][4]*NUMV[('insulin','rate')]['iqr']+NUMV[('insulin','rate')]['median']
    rec=rollout(W0,B_,[None,(8,2.0,12.0,13.0)],npaths=20,seed=3)
    prop.append([float((rec[(h_,6)][1]-rec[(h_,6)][0]).mean())/d0 if (h_,6) in rec else np.nan for h_ in range(13,25)])
prop=np.array(prop); print(len(prop), 'single-dose rollout IRF per U/h, hours 13..24:', np.round(np.nanmean(prop,0),3))
```