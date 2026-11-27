VOCAB has no per-drug key at the top level. The rate normalisation and bin edges should be inside VOCAB['numeric'], explained by 'numeric_note'.

The fwd_11 CSV holds per stay-hour shifts with a 'reached' flag, so I can get the hourly profile. This run will:
- print the numeric note and norepinephrine's rate entry (bin edges and normalisation);
- print the hourly mean norepinephrine shifts for the edited and original models.

```python
print(str(VOCAB['numeric_note'])[:400])
nu = VOCAB['numeric']; print(type(nu).__name__, (list(nu.keys())[:10] if isinstance(nu, dict) else len(nu)))
ne_ent = nu.get('norepinephrine') if isinstance(nu, dict) else [x for x in nu if 'norepinephrine' in str(x)]
print(str(ne_ent)[:700])
for nm_, df_ in [('ed', de), ('or', do)]:
    for e_ in ['ne_x2','ne_half']:
        s_ = df_[(df_.edit==e_) & (df_.reached.astype(bool))].groupby('hour')['shift'].mean()
        print(nm_, e_, np.round(s_.values, 2).tolist())
```