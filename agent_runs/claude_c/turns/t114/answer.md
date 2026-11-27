Pass 10 settles the z question. Amplifying the z token x4 along both the glucose- and MAP-surprise directions left the factual results unchanged: NLL 1.63461 against 1.63462, with identical errors for every event. The state update ignores the z token, so the z route is dead. Together with the earlier finding that state-space directions are not selective, that leaves the action token u_k as the only drug-specific channel that actually moves the state.

The norepinephrine problem may be one of direction rather than size. Norepinephrine is titrated on past MAP, so a large dose signals a patient who needs pressor support. The model may read "more norepinephrine" as "sicker patient", which predicts a lower MAP. My notes say the wrong sign comes from stays capped at 1.0, where x2 is out of distribution.

Before choosing an edit inside the action encoder, I want the free per-stay-hour breakdown of the existing dose-edit outputs:
- original (pass 1) and dose columns x5 (pass 4);
- split by whether the stay's norepinephrine at hour 12 or later is at the cap;
- for insulin, split by dose level.

If non-capped stays already show right-signed but small norepinephrine responses, amplifying the existing pathway is the right move for both drugs and the cap needs a separate fix. If even non-capped stays are wrong-signed, norepinephrine needs a new pathway.

```python
de1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de1.shape, list(de1.columns)); print(de1.head(3).to_string())
print(de1.iloc[:, :3].drop_duplicates().shape if de1.shape[1]>2 else '')
for c in de1.columns:
    if de1[c].dtype==object: print(c, de1[c].unique()[:8])
```