The per-draw head inputs extract correctly, and they rule out my diagnosis. Per draw, the readouts vary by only 0.08–0.09 across z-draws. The gates separate on every draw: the arterial-versus-cuff gate cleanly, and the insulin gate with only a hairline overlap at the extremes. The glucose-versus-MAP code gate is clean too.

So the huge MAP shifts do not come from noisy inputs. Two causes remain:
- the units do not compute what I think;
- the size of the logit tilt is too large, so the mean is no longer linear in it.

Next I will run the gated edit and read off the four units' activations per draw, using the head's fourth output. I will check whether u1 − u2 equals the insulin readout inside the gate and 0 outside, and likewise u3 − u4 for norepinephrine. I will also measure how the predicted mean responds to the size of the logit tilt along `tg` and `tm`.

If the unit differences are right but the tilt response is strongly nonlinear, the logit direction needs rescaling. If the unit differences are wrong, the construction itself has a bug.

```python
Xs_,R_=datasets[0]
oE=head_run(Xs_,R_,Eg); H_=npv(oE[3]).astype(np.float64)
u1,u2,u3,u4=units
di=(H_[:,:,u1]-H_[:,:,u2]); dn_=(H_[:,:,u3]-H_[:,:,u4])
ri_=WCd0@ROi['w']+ROi['w0']; rn_=WCd0@ROn['w']+ROn['w0']
for nm,dd,rr,groups_ in [('ins pair',di,ri_,[(6,True),(6,False),(10,True),(10,False),(7,None),(5,None)]),('ne pair',dn_,rn_,[(7,None),(5,None),(6,None),(10,None)])]:
    for c,h in groups_:
        m_=(cd0==c) if h is None else ((cd0==c)&(hi0==h))
        e_=dd[:,m_]-(rr[:,m_] if (nm=='ins pair' and c in (6,10) and h) or (nm=='ne pair' and c==7) else 0)
        print(nm,'code',c,'hi',h,'n',m_.sum(),'pair out pct1/50/99 %s | err vs target maxabs %.3f'%(np.round(np.percentile(dd[:,m_],[1,50,99]),2),np.abs(e_).max()))
print('pre-unit check: max |hidden| others %.2f'%np.abs(np.delete(H_,units,axis=2)).max())
```