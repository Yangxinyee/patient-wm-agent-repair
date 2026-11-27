`M2` was built with the first residual builder, whose lag columns are named differently: `ei_l`k / `pol_l`k for insulin deviation / policy, and `en_l`k / `nexp_l`k / `nor_l`k for norepinephrine. The test itself is unchanged; I'll rerun the same regressions with the right names.

```python
def show(tab): 
    t,n,s=tab; print('n',n,'sd',round(s,2)); print(t.T.to_string())
gfm=M2[M2.code==CODE['glucose_finger']]; mam=M2[M2.code==CODE['map_arterial']]
show(ols_tab(gfm,'res',['ei_l1','ei_l2','ei_l3','ei_l4','pol_l1','pol_l2','pol_l3','pol_l4']))
show(ols_tab(gfm,'res',['ins_l1','ins_l2','ins_l3','ins_l4','pol_l1','pol_l2','pol_l3','pol_l4']))
show(ols_tab(mam,'res',['en_l1','en_l2','en_l3','en_l4','nexp_l1','nexp_l2','nexp_l3','nexp_l4']))
show(ols_tab(mam,'res',['nor_l1','nor_l2','nor_l3','nor_l4']))
```