The bundles match the records, and END atoms are separate atoms. At hour 1.083 the bundle holds 4 atoms: the new insulin and norepinephrine records plus the ENDs of the previous hour's two records. At 2.083 it holds 3, because norepinephrine stopped. Measurements and the ICU admission at hour 0 form one 3-atom bundle.

Next I will rebuild the bundle and atom lists for the 12 stays from the tuning records. Each list has the code, kind, event kind, value, rate and dose of every atom, with END atoms placed at end_hour. I will check the counts against the states file. If every stay's bundle hours and atom counts match exactly, the input side is settled and I can move to the embedding and encoders.

```python
code_id = {c_['name']: c_['id'] for c_ in VOCAB['codes'] if 'name' in c_}
ek_id = {'measurement':1,'drug_admin':2,'visit':3,'visit_detail':4}
code_ek = {c_['name']: ek_id.get(c_.get('event_kind'),0) for c_ in VOCAB['codes'] if c_.get('event_kind')}
def build_bundles(rec):
    at = []
    for r_ in rec.itertuples(index=False):
        c_ = code_id[r_.event]; e_ = code_ek[r_.event]
        if e_==2:
            at.append((round(r_.hour*60), c_, 0, e_, np.nan, r_.rate, r_.dose))
            at.append((round(r_.end_hour*60), c_, 2, e_, np.nan, r_.rate, r_.dose))
        else:
            at.append((round(r_.hour*60), c_, 0, e_, r_.value if e_==1 else np.nan, np.nan, np.nan))
    A_ = pd.DataFrame(at, columns=['minute','code','kind','ekind','value','rate','dose']).sort_values('minute', kind='stable')
    return A_
stays12 = list(np.unique(F4['stay_id']))
BUN = {}
mism = 0
for s_ in stays12:
    A_ = build_bundles(tu[tu.stay_id==s_].sort_values('hour', kind='stable'))
    mins = sorted(A_.minute.unique())
    mk_ = F4['stay_id']==s_
    hs = np.round(F4['hour'][mk_]*60).astype(int)
    na = A_.groupby('minute').size().reindex(mins).values
    ok_ = (len(mins)==mk_.sum()) and np.all(np.array(mins)==hs) and np.all(na==F4['n_atoms'][mk_])
    if not ok_:
        mism += 1
        print(s_, len(mins), mk_.sum(), mins[-3:], hs[-3:].tolist(), na[-3:].tolist(), F4['n_atoms'][mk_][-3:].tolist())
    BUN[s_] = A_
print('mismatching stays', mism)
```