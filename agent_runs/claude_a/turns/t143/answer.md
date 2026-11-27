Fitted on drug-start bundles alone, head 3 decodes the insulin rate almost perfectly (R² 0.997, residual sd 0.06). σ_x is tight within bundle type (≈80–86 at insulin bundles, dt always 300 s), so dividing by the LayerNorm scale costs only a few percent.

Off-target, though, that decoder gives huge values: −1,114 ± 571 at reading bundles, where head 3 reads the e token instead of u. The rate readout must also be near zero on e and slot tokens. Next I refit with the constraint included: target = insulin rate at drug bundles and 0 at every other bundle, with off-target rows upweighted. I compare head 3 alone with heads 2+3, and raw versus σ-scaled features.

If some fit keeps the residual ≲0.1 both at insulin bundles and off-target, a plain linear staging (no gating) works. If not, I need an FFN gate keyed on bundle type in the prelude.

```python
def wfit_eval(XX, y, w_off=1.0, lab=''):
    wt=np.where(drug_b,1.0,w_off)
    Xa=np.c_[XX,np.ones(N2)]
    sw=np.sqrt(wt)[:,None]
    w_,_,_,_=np.linalg.lstsq(Xa*sw, y*sw[:,0], rcond=None); p_=Xa@w_
    s_=f"{lab:22s} ins {np.std(p_[ins_s]-y[ins_s]):.3f} ne_only {np.std(p_[ne_s&~ins_s]-y[ne_s&~ins_s]):.3f}"
    for nm_ in ['end_only','gluc_read','map_only','other']:
        m_=types[nm_]; s_+=f" | {nm_} {np.mean(p_[m_]-y[m_]):+.3f}±{np.std(p_[m_]-y[m_]):.3f}"
    print(s_); return w_,p_
y_r=rate_ins.copy()
X23r=A_[:,:4,2:4,:].mean(1).reshape(N2,96).numpy()
X23s=(A_[:,:4,2:4,:]/sdx[:,:,None,None]).mean(1).reshape(N2,96).numpy()
for lab_,XX in [('h3 raw',X3r),('h3 /sig',X3),('h23 raw',X23r),('h23 /sig',X23s)]:
    for wo_ in [1.0,10.0]:
        wfit_eval(XX,y_r,wo_,f"{lab_} woff={wo_}")
```