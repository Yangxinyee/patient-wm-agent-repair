The rollout crashed on a dtype mismatch. The memory ages in the recorded prefix were built from numpy int64 minutes and became float64 tensors. I'll make `age_emb_P` cast its input to float32 and rerun the same validation against forward 1.

```python
def age_emb_P(age_s, P):
    age_s = torch.as_tensor(age_s, dtype=torch.float32)
    return F.linear(F.gelu(F.linear(torch.log1p(age_s)[:,None], P['core.age.0.weight'], P['core.age.0.bias'])), P['core.age.2.weight'], P['core.age.2.bias'])
t0 = time.time(); RO = pd.concat([rollout_stay(tu, s, Po, n_paths=20, seed=k_) for k_, s in enumerate(sel1[:30])]); print("time", round(time.time()-t0,1))
cmp_ = RO.merge(D1f.rename(columns={'shift':'shift_tool','factual':'fact_tool','reached':'reached_tool'}), on=['stay_id','hour','edit'], how='inner')
print("rows matched", len(cmp_), "of", len(RO), "| reached agree", float((cmp_.reached==cmp_.reached_tool).mean()))
for ed, g_ in cmp_.groupby('edit'):
    rr = g_[g_.reached_tool]
    print(ed, "fact rmse", round(float(np.sqrt(((g_.factual-g_.fact_tool)**2).mean())),2), "| reached n", len(rr), "mine", round(rr['shift'].mean(),3), "tool", round(rr['shift_tool'].mean(),3), "corr", round(np.corrcoef(rr['shift'], rr['shift_tool'])[0,1],2) if len(rr)>2 else None)
```