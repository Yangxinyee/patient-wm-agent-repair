Run 60 lists the tensors. The dose information can only reach the glucose prediction through the action encoder, the posterior `z_post` (which reads the action vector u_k as input columns 384 to 575) and the prelude token. Of these, the 32-dimensional z posterior is the only one I can observe, and z feeds the state through `z_in.weight` (192×32).

My notes say that at insulin bundles only z dimension 26 varies. Before spending a forward pass, I will check this more fully with the saved states from pass 5:
- the variance of each z dimension at insulin, norepinephrine and reading-only bundles;
- how much of the z26 variation at insulin bundles tracks dose, and how much the edited model (z_dose/z_rate inputs ×5) changed it.

If dose sits in one or two z dimensions, then amplifying `z_in` along those dimensions is a cheap, testable lever. The obstacle would be large variance in the same dimensions at reading-only bundles, because amplifying them would then disturb the factual fit.

```python
print(open('/data/README.md').read()[:2500])
So_=np.load('/work/fwd_5/states_original.npz'); Se_=np.load('/work/fwd_5/states_edited.npz')
print(So_.files)
xo_=So_['x']; xe_=Se_['x']
hi_=So_['has_insulin'].astype(bool); hn_=So_['has_norepinephrine'].astype(bool)
zo_=xo_[:,960:992]; ze_=xe_[:,960:992]
grp_={'ins_only':hi_&~hn_,'ne_only':hn_&~hi_,'both':hi_&hn_,'read':~hi_&~hn_}
for k_,msk in grp_.items():
    print(k_, msk.sum(), 'var>0.05 dims:', [(j,round(float(zo_[msk,j].var()),2)) for j in range(32) if zo_[msk,j].var()>0.05])
dz_=ze_-zo_
for k_,msk in grp_.items():
    print('edit-orig dz', k_, [(j,round(float(np.abs(dz_[msk,j]).mean()),3)) for j in range(32) if np.abs(dz_[msk,j]).mean()>0.02])
```