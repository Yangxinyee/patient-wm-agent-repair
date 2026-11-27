Run 106 shows the gap scaling log(seconds)/10 is correct, since the alternatives did worse. The best-correlating type, 1, matches the encoding type = atom kind × 5 + event kind, which gives 1 for a measured value. So I'll fix type 1. The remaining error (bias −7.4 mg/dL, too little spread) should come from the register block. Next I'll vary the register inputs one at a time with type 1: none at all, age in hours, the bin edge convention, and the treatment of the "no value" bin. Whichever change cuts the error most points to the missing piece.

```python
def headv(df,typ=1,**kw): return head(df,typ,**kw)[0]
base_=headv(gfm); print('base',round(np.mean(np.abs(base_-pm)),2),round((base_-pm).mean(),2))
v=gfm.copy(); v['has_last']=0; m_=headv(v); print('noreg',round(np.mean(np.abs(m_-pm)),2),round((m_-pm).mean(),2))
v=gfm.copy(); v['age_s']=gfm.age_s/3600; m_=headv(v); print('age_h',round(np.mean(np.abs(m_-pm)),2),round((m_-pm).mean(),2))
m_=headv(gfm,side='left'); print('left',round(np.mean(np.abs(m_-pm)),2))
m_=headv(gfm,renorm=False); print('norenorm',round(np.mean(np.abs(m_-pm)),2))
m_=headv(gfm,regv=2); print('regv2',round(np.mean(np.abs(m_-pm)),2))
```