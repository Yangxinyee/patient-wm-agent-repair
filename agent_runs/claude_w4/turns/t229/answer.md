v3 is clean on two of three counts. At drug bundles the original head-1 values have essentially no component along the three injection directions (sd ≈ 0.001), so the remapping does not disturb the original computation. The insulin readout `tI` tracks the recorded insulin rate well (rms error 0.11 units/h) and is ≈0 when insulin is absent. The presence readout `pI` is a clean 0/1 indicator.

The NE readout `fN`, however, is off by 0.5 rms against `RN` on a 0–1 scale. Either `fN` is on a different scale, for example a standardised rate, or it fails on some bundle types, such as bundles carrying both drugs. The size check also fits: the injection is small relative to the slot state (|dI| = 13 per unit/h against slot norms near 800), so it should act as a gentle perturbation.

Next I'll diagnose `fN`. I'll regress it on `RN` overall and separately for NE-only bundles and bundles with both drugs, and compare it with the standardised rate (r−0.47)/0.6. I'll also try to obtain the no-action token to confirm that the injection is silent at measurement bundles.
- **Simple fix:** if `fN` is an affine rescaling of the rate, I can fold that into gN and the offset.
- **Refit needed:** if it fails on bundles with both drugs, I'll refit an NE-rate readout on the cached tokens before forward-testing NE.

```python
pres=RN>0
for nm,msk in [('all present',pres),('N only',pres&(RI==0)),('both',pres&(RI>0))]:
    if msk.sum()<5: print(nm,'n',msk.sum()); continue
    A_=np.c_[RN[msk],np.ones(msk.sum())]; coef,res,_,_=np.linalg.lstsq(A_,fNr[msk],rcond=None)
    e=fNr[msk]-A_@coef
    print(f'{nm}: n {msk.sum()} fN = {coef[0]:.4f}*rate + {coef[1]:+.4f}  resid rms {np.sqrt((e**2).mean()):.4f}; corr {np.corrcoef(RN[msk],fNr[msk])[0,1]:.4f}; vs z=(r-.47)/.6 rms {np.sqrt(((fNr[msk]-(RN[msk]-.47)/.6)**2).mean()):.4f}')
print('fN on absent-NE rows: mean %.4f sd %.4f'%(fNr[~pres].mean(),fNr[~pres].std()))
# what other NE readouts exist?
for dn,dct in [('BB',BB),('RO',RO),('TT',TT),('bsel',bsel)]:
    for k,v in dct.items():
        v=np.asarray(v)
        if v.shape==(192,):
            r_=Xu@v; t_=RN if 'N' in k else RI; m_=t_>0
            e=r_-t_; print(f'{dn}[{k}]: rms vs rate all {np.sqrt((e**2).mean()):.4f} present {np.sqrt((e[m_]**2).mean()):.4f}')
for fnm in ['u_set','U_of']:
    try:
        u0=np.asarray(globals()[fnm]([]),dtype=np.float64).ravel(); x0=LN1(u0)
        print(fnm,'([]) ok norm %.3f; orig proj on aI3,aN3,aP3:'%np.linalg.norm(u0), np.round((x0@V1.T)@np.stack([aI3,aN3,aP3]).T,4), ' readouts tI %.4f fN %.4f pI %.4f'%(float(x0@BB3['tI']),float(x0@BB3['fN']),float(x0@BB3['pI'])))
        break
    except Exception as e: print(fnm,'failed',e)
```