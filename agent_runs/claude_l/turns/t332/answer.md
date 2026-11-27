The out-of-sample factual check passed comfortably. On 100 tuning stays never used for fitting, the edited model's value NLL is **1.68915 against 1.69137 for the original**, so −0.0022 (better), against a limit of +0.005.
- **Glucose:** finger costs +0.0031 per reading and lab +0.0007. This is the insulin correction.
- **MAP:** arterial improves by −0.0048 and cuff by −0.0056. This comes from the deterministic head, and possibly also from the norepinephrine correction.

There is therefore about 0.007 of headroom. I can raise the insulin gain, which drives the main readout, well above the current 0.3 of full scale. A pessimistic purely quadratic model of the glucose cost puts insulin scale 0.7 at a total of about +0.002 against the original.

This is my last Python run, so it must choose conservatively. It does the following:
1. Evaluates four candidates offline, each with the batch-1 in-sample NLL cost and the rollout readouts: (insulin, norepinephrine) scales (0.5, 0.33), (0.6, 0.35), (0.7, 0.35) and (0.8, 0.35).
2. Predicts each one's out-of-sample total. It starts from the measured −0.0022 and adds 1.3 times the in-sample cost growth over the current setting.
3. Saves the largest candidate whose predicted total is at most +0.0025, which leaves a 0.0025 margin to the limit. If none qualifies, the current edit stays.
4. Writes the final summary to the notebook.

How I will read the outcome:
- **Confirms:** the readouts grow roughly linearly with the insulin scale, towards −10 to −11 for insulin ×2, while the in-sample cost grows slowly.
- **Refutes:** a sharp jump in cost, in which case a smaller candidate gets saved.

```python
import numpy as np, torch, time
t0=time.time(); OUT=[]; chosen=(0.3,0.3,None)
try:
    tb=tfnll(W_cur,SL)[0]; C3=0.0022; best=None
    for fi,fn in [(.5,.33),(.6,.35),(.7,.35),(.8,.35)]:
        Wt=mkWz(bI*fi,bN*fn,Q=Qall); c=tfnll(Wt,SL)[0]-tb; R={k:round(v[0],2) for k,v in evalW(Wt,SL).items()}
        D=-0.0022+1.3*(c-C3); OUT.append((fi,fn,round(c,4),round(D,4),R)); print(fi,fn,'inS dNLL %.4f predD %.4f'%(c,D),R,round(time.time()-t0))
        if D<=0.0025: best=(fi,fn,Wt,R)
    if best is not None:
        fi,fn,Wt,R=best
        for nme in ['z_prior.0.weight','z_prior.0.bias','z_prior.2.weight','z_prior.2.bias','z_head.weight']: save_edit(nme,Wt[nme].float().contiguous())
        chosen=(fi,fn,R); print('SAVED fi %.2f fn %.2f gains %.2f %.2f'%(fi,fn,bI*fi,bN*fn),R,edited())
    else: print('kept fi .3 fn .3')
except Exception as e:
    import traceback; print('ERR',str(e)[:200]); print(traceback.format_exc()[-300:])
with open('/work/notes.md','a') as f:
    f.write(f"""
## FINAL (run 300)
- fwd_6 factual both, 100 NEW tuning stays, edit fi=fn=.3: edited 1.68915 vs original 1.69137 (-0.0022); finger +.0031, lab +.0007, art -.0048, cuff -.0056.
- Run 300 candidates (fi,fn,inSample dNLL vs W_cur,pred total vs orig,readouts): {OUT}
- SUBMITTED: fi={chosen[0]}, fn={chosen[1]} (gains bI*fi, bN*fn with bI={bI:.2f}, bN={bN:.2f}); offline batch1 readouts {chosen[2]} vs IRF truth -14.2/7.1/4.9/-2.45.
- Edit = z_prior rebuilt as deterministic linear pass-through (gelu linear regime K=12, logsig -30) of two ridge decoders on the view base
  (targets: previous-bundle dose residual vs ctx policy; + penalty matching response to rollout dose-doubling Delta); z_head cols 0/1 = gain x
  glucose-only / MAP-only view directions (head Jacobians). Everything else original (z_head other cols 0).
- Belief: true effects ~3-4x the original model's; residual-keyed head correction amplifies lag-1 dose effect; rollout persistence accumulates it.
  Limits: model has no dose memory, so the decoder is noisy; gains capped by factual NLL; NE response symmetric (log-like) -> x2 under, x1/2 ok.
""")
```