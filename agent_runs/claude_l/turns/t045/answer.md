The kernel still holds my data objects:
- `P`: the hourly training panel, 30,000 stays × 26 hours.
- `PT`: the same panel for the 400 tuning stays.
- `base_df`: the insulin decision rows with residual `ires`.
- `gb` / `gbn`: the policy GBMs for insulin and norepinephrine.

Function sources are lost, so I will rebuild what I need in a clean, documented form.

Two stay lists exist: `batch1` (used for the dose-edit pass 1) and `stays_b1` (probably the stays in the pass-2 states). I need to know whether these sets coincide before pairing states with any factual pass. I also need the exact policy feature list and the dose-edit CSV layout. So this run is purely for inventory: the README, the panel indexes, the feature lists, the stay sets and the CSV columns.

```python
print(open('/data/README.md').read()[:3000])
print(P.index.names, P.head(3)); print(PT.index.names, PT.head(3))
print('X_cols', X_cols, 'gb nfeat', gb.n_features_in_, 'gbn nfeat', gbn.n_features_in_)
S2 = np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
u2 = list(dict.fromkeys(S2['stay_id'].tolist()))
print('fwd2 stays', len(u2), u2[:5], 'same as batch1:', set(u2)==set(batch1), 'same as stays_b1:', set(u2)==set(stays_b1))
de1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(de1.columns.tolist()); print(de1.head(3))
print(json.load(open('/work/fwd_1/summary.json')))
```