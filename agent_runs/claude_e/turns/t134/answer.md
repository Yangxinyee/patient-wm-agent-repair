The gated design is right in structure but fails in practice for norepinephrine.

| Rows | dNLL | Shift |
|---|---|---|
| Removing the four units (a = 0) | −0.001 overall | — |
| Finger, outside insulin gate | +0.0016 | about 0 |
| Finger, inside insulin gate | +0.0026 | slope −2.19 |
| Arterial and cuff MAP | +17 to +38 | about −17 mean, SD 15–22 |
| Lab glucose, inside insulin gate | +0.028 | SD 14 |

So removing the four units costs nothing, and on finger glucose the gate works: almost free, with the intended slope. On the MAP codes and lab glucose the edit is broken. Inside a working gate the output is bounded by the readout's range, so the broken units must be getting inputs far outside the range I fitted on.

The likely cause is that I fitted both the gates and the readouts on z-averaged head inputs. The head runs each z-draw separately. The weakly regularised readout (×5 for unit gain on norepinephrine) and the discriminants can then pick up directions where single draws vary a lot.

The fix is to fit on per-draw inputs. I can extract them with the existing head code: an identity first layer with bias +100 makes the hidden layer return input + 100 exactly. Next I will extract them for dataset 0 and confirm that their draw-average reproduces the stored averages. Then I will measure how widely the current readouts and gates vary across draws.

If the per-draw spread of the norepinephrine readout or of the arterial-versus-cuff gate is far larger than their averaged-input range, that confirms the diagnosis.

```python
Wid=dict(W0=torch.eye(192,dtype=torch.float64),b0=torch.full((192,),100.,dtype=torch.float64),W2=torch.zeros(17,192,dtype=torch.float64),TR=ORIG['TR'].clone())
def wc_draws(i):
    Xs_,R_=datasets[i]; o=head_run(Xs_,R_,Wid); return npv(o[3]).astype(np.float64)-100.
WCd0=wc_draws(0); n0=WCd0.shape[1]
print('per-draw wc', WCd0.shape, 'mean-over-draws vs ALLW[0] maxdiff %.2e'%np.abs(WCd0.mean(0)-ALLW[0]).max())
cd0=A_.code.values[:n0]; hi0=A_.hi.values[:n0]
ri=WCd0@ROi['w']+ROi['w0']; rn=WCd0@ROn['w']+ROn['w0']
for nm,r_,m_ in [('ins readout glucose&hi',ri,np.isin(cd0,[6,10])&hi0),('ins readout code10&hi',ri,(cd0==10)&hi0),('ne readout code7',rn,cd0==7)]:
    print(nm,'avg-input sd %.2f | per-draw sd across draws (mean) %.2f | per-draw pct 0.5/99.5 %s'%(r_.mean(0)[m_].std(),r_[:,m_].std(0).mean(),np.round(np.percentile(r_[:,m_],[0.5,99.5]),1)))
for nm,(w_,c__,s__),m1,m2 in [('ins gate',gi_,np.isin(cd0,[6,10])&hi0,np.isin(cd0,[6,10])&~hi0),('code7 gate',g7_,cd0==7,cd0==5)]:
    p_=(WCd0@w_-c__)/s__
    print(nm,'per-draw: pos 0.1%% %.2f min %.2f | neg 99.9%% %.2f max %.2f | draw sd %.2f'%(np.percentile(p_[:,m1],0.1),p_[:,m1].min(),np.percentile(p_[:,m2],99.9),p_[:,m2].max(),p_.std(0).mean()))
pg_=WCd0@gvn
print('gvec per-draw by code:',{c:(round(pg_[:,cd0==c].min(),2),round(pg_[:,cd0==c].max(),2)) for c in [5,6,7,10]})
```