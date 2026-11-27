Notebook first. Session 6 is not recorded, so I am writing down what I reconstructed. The diag output shows vI and vN have unit response to their own drug's edits. vI also leaks about -0.3 to the NE x2 edit, and vN has almost no leakage. The noise figures (15.4/7.4 for vI, 3.3/0.94 for vN) are unexplained, so next I need their units. I will regress each decoder's edit response on the lagged dose changes. That shows which lag-weighted dose quantity the decoders track, and puts the factual noise in the same units.

```python
with open('/work/notes.md','a') as fh:
    fh.write("""
## Session 6 (runs 192-199, reconstructed; runs 166-191 were not logged)
- Kernel has Xf(70738x192 head inputs, 1500 train stays), cf code id (0 finger,1 lab,2 art,3 cuff), lbf lastbin, Rf=cell-residualised Xf,
  DQf dose-direction derivs, EDR[(drug,factor)]={gi,dX,dd(4 lags),reached} replica edit responses; solve_unit -> decoders vI (glucose rows), vN (MAP rows).
- diag: vI noise 15.4/7.4, own response 1.0, leak NEx2 -0.31; vN noise 3.3/0.94, own 1.0, leak ~0. Units being checked.
""")
for key,v_,code_ok in [(('insulin',2.0),vI,[0,1]),(('norepinephrine',2.0),vN,[2,3])]:
    e_=EDR[key]; m_=np.isin(cf[e_['gi']],code_ok)&e_['reached'].astype(bool)
    y_=e_['dX'][m_]@v_; D_=e_['dd'][m_]
    b_=np.linalg.lstsq(np.c_[D_,np.ones(len(D_))],y_,rcond=None)[0]
    print(key,'n',m_.sum(),'coef lags1-4',np.round(b_[:4],3),'icpt',round(b_[4],3),'mean resp',round(y_.mean(),3),'mean dd1',round(D_[:,0].mean(),3))
    rw=ins_rows if code_ok[0]==0 else map_rows
    print('  factual sd of v.Rf on rows',round(np.std(Rf[rw]@v_),3))
```