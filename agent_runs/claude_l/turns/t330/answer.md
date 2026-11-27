Run 298 showed that the gated variant is unsafe, so I am reverting it:
- **The gates themselves work.** They separate cleanly, about 98–100% on held-out stays.
- **Insulin cost exploded.** The held-out NLL cost was 0.096 at full gain, against 0.015 without the gate.
- **Saved edit is over budget.** It was saved at gains 0.2 and 0.35 and costs +0.0076 in-sample. Even allowing for the 0.0015 that zeroing `z_head` gained, that is about +0.006 against the original model, over the 0.005 limit.
- **Likely cause.** The gated decoder is fit only on post-drug points, so elsewhere its output is unconstrained. The 1.5–2.5% of points where the gate leaks then get large, wrong corrections.

So I revert to the ungated decoder from run 297, which behaved predictably (in-sample cost +0.0017 at gains 0.2 and 0.3). I use gains 0.3 and 0.3, the best allocation for a held-out budget of 0.0045. Predicted readouts are about −5.3 / +3.1 / +1.6 / −2.0, against the truth of −14.2 / +7.1 / +4.9 / −2.45. `batches.json` holds only batch 1, so I also list the unused tuning stays. The tool can then measure the true out-of-sample factual cost.

How I will read the outcome:
- **Confirms:** the saved tensors match the intended ones, the in-sample cost is about 0.002–0.003, and the readouts are near the prediction.
- **Then:** if the tool's factual check on new stays shows the NLL within about +0.003 of the original, the edit is safe. My last Python run can then fine-tune the gains.

```python
import numpy as np, torch, pandas as pd, time
t0=time.time()
try:
    fi,fn=0.3,0.3; Wf=mkWz(bI*fi,bN*fn,Q=Qall)
    for nme in ['z_prior.0.weight','z_prior.0.bias','z_prior.2.weight','z_prior.2.bias','z_head.weight']: save_edit(nme,Wf[nme].float().contiguous())
    mx=max(float((torch.as_tensor(load(n)).double()-Wf[n].double()).abs().max()) for n in ['z_prior.0.weight','z_prior.2.bias','z_head.weight'])
    rf=evalW(Wf,SL); tff=tfnll(Wf,SL)[0]; tb=tfnll(W_cur,SL)[0]
    print('SAVED ungated gains %.2f %.2f'%(bI*fi,bN*fn),'maxdiff %.1e'%mx,{k:round(v[0],2) for k,v in rf.items()},'in-sample dNLL %.4f'%(tff-tb),edited(),round(time.time()-t0))
    with open('/work/notes.md','a') as f:
        f.write(f"- Run 298 gated variant REJECTED: in-sample dNLL .0076 (gate leakage + off-distribution decoder). Run 299: re-saved UNGATED mkWz(Qall) gains {bI*fi:.2f},{bN*fn:.2f} (fi=fn=.3, HO-predicted cost .0044 vs W_cur); offline readouts {dict((k,round(v[0],2)) for k,v in rf.items())} truth -14.2/7.1/4.9/-2.45; in-sample dNLL {tff-tb:.4f}. To rescale: z_head[:,0]=bI'*vg, z_head[:,1]=bN'*vm (q9.npz). Next: tool factual both on new tuning stays.\n")
except Exception as e:
    import traceback; print('ERR',str(e)[:200]); print(traceback.format_exc()[-400:])
try:
    tu=pd.read_csv('/data/tuning.csv'); sc=[c for c in tu.columns if 'stay' in c.lower()][0]
    ids=[x for x in pd.unique(tu[sc]) if str(x) not in set(map(str,SL))]
    print('col',sc,'new tuning ids',len(ids)); print(','.join(str(x) for x in ids[:100]))
except Exception as e: print('ids err',str(e)[:150])
```