`truth_ins` and `truth_ne` take a stay, an hour and a dose multiplier. They may read tables I reassigned this session: I rebuilt `ins` and `ne` from the same tuning records, with the same shapes as before. So first I re-derive the f3 truth from the fwd_16 rows and check it against my notes (insulin x2 ≈ −13.18, x1/2 ≈ +6.59, NE x2 ≈ +4.76, NE x1/2 ≈ −2.38). If it matches, I compute the tu1 truth for fwd_22 and compare it with the edited model's readouts.

```python
def truth_tab(df, k):
    out_={}
    for ed,fn,mu in [('insulin_x2',truth_ins,2.0),('insulin_half',truth_ins,0.5),('ne_x2',truth_ne,2.0),('ne_half',truth_ne,0.5)]:
        r=df[(df.edit==ed)&(df.reached)]
        t=np.array([fn(s,h,mu) for s,h in zip(r.stay_id,r.hour)],dtype=float)
        out_[ed]=(round(float(np.nanmean(t)),3), round(float(r['shift'].mean()),3), len(r))
    print(k, out_)
e16_=pd.read_csv('/work/fwd_16/dose_edit_edited.csv'); truth_tab(e16_,'f3 (fwd_16, -2.5): (truth, edited, n)')
truth_tab(e22,'tu1 (fwd_22, -2.6): (truth, edited, n)')
```