This is still the interface inventory, so it does not resolve the END-atom mismatch. I’ll now run the reconstruction test itself. A near-zero posterior error would validate this pathway; otherwise I’ll avoid basing an edit on the approximate reconstruction.

```python
import itertools
import torch.nn.functional as tF

Wrepair = {n: load_original(n) for n in NAMES}
sx_repair = torch.tensor(S_reference["x"], dtype=torch.float32)
sm_repair = pd.DataFrame({k: S_reference[k] for k in
    ("stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms")})
start_lookup_repair, end_lookup_repair = {}, {}
for row in reference_records_repair.to_dict("records"):
    key = (int(row["stay_id"]), round(float(row["hour"]), 4))
    start_lookup_repair.setdefault(key, []).append(row)
    if row["event"] in ("insulin", "norepinephrine"):
        key = (int(row["stay_id"]), round(float(row["end_hour"]), 4))
        end_lookup_repair.setdefault(key, []).append(row)

empty_e_repair = encode_atom_set([], "encode").reshape(1, 192)
action_cases_repair = []
for j in range(1, len(sm_repair)):
    if sm_repair.iloc[j]["stay_id"] != sm_repair.iloc[j-1]["stay_id"]:
        continue
    key = (int(sm_repair.iloc[j]["stay_id"]), round(float(sm_repair.iloc[j]["hour"]), 4))
    starts, ends = start_lookup_repair.get(key, []), end_lookup_repair.get(key, [])
    if starts and all(r["event"] in ("insulin", "norepinephrine") for r in starts):
        if len(starts) + len(ends) == int(sm_repair.iloc[j]["n_atoms"]):
            action_cases_repair.append((j, starts, ends))

def previous_context_repair(indices):
    ii = np.asarray(indices, dtype=np.int64)
    dt = torch.tensor((sm_repair.iloc[ii]["hour"].to_numpy()
        - sm_repair.iloc[ii-1]["hour"].to_numpy())*3600, dtype=torch.float32)
    old = sx_repair[ii-1]
    hf = old[:,192:960].reshape(-1,4,192) * torch.exp(
        -dt[:,None,None]*torch.exp(Wrepair["core.flow.a"]))
    ctx = old[:,:192] + rnorm(hf.mean(1),"core.out_norm")
    return ctx + rmlp(torch.log1p(dt).reshape(-1,1)/10,"gap"), hf

sc = [c for c in action_cases_repair if len(c[1])==1 and not c[2]][:100]
ctx, _ = previous_context_repair([c[0] for c in sc])
u = action_encode_atoms(torch.stack([embed_record_atom(c[1][0]) for c in sc])[:,None,:])
zh = rmlp(torch.cat([ctx,empty_e_repair.expand(len(sc),-1),u],1),"z_post")[:,:32]
print("Single-start count/RMSE:",len(sc),
      float(((zh-sx_repair[[c[0] for c in sc],960:992])**2).mean().sqrt()))

ec = [c for c in action_cases_repair if len(c[1])==len(c[2])==1
      and c[1][0]["event"]==c[2][0]["event"]][:100]
ctx, _ = previous_context_repair([c[0] for c in ec])
target = sx_repair[[c[0] for c in ec],960:992]
starts = torch.stack([embed_record_atom(c[1][0]) for c in ec])
print("Unit IDs:",unit_ids_repair,"Rate-unit IDs:",rate_unit_ids_repair)
print("Kinds:",VOCAB["atom_kinds"],VOCAB["event_kinds"])

end_search_repair = []
for dm,rm,uz,ruz,ez in itertools.product(
    ("recorded","absent","zero"),("recorded","absent","zero"),
    (False,True),(False,True),(False,True)):
    aa = []
    for c in ec:
        row = dict(c[2][0])
        if dm!="recorded": row["dose"] = np.nan if dm=="absent" else 0.
        if rm!="recorded": row["rate"] = np.nan if rm=="absent" else 0.
        a = embed_record_atom(row,kind1=2,end_mode1="same")
        event = row["event"]
        if uz:
            w = Wrepair["embed.categorical.unit.weight"]
            a = a-w[unit_ids_repair[event]]+w[0]
        if ruz:
            w = Wrepair["embed.categorical.rate_unit.weight"]
            a = a-w[rate_unit_ids_repair[event]]+w[0]
        if ez:
            w = Wrepair["embed.ekind.weight"]
            a = a-w[2]+w[0]
        aa.append(a)
    u = action_encode_atoms(torch.stack([starts,torch.stack(aa)],1))
    zh = rmlp(torch.cat([ctx,empty_e_repair.expand(len(ec),-1),u],1),"z_post")[:,:32]
    end_search_repair.append((dm,rm,uz,ruz,ez,float(((zh-target)**2).mean().sqrt())))
end_search_table_repair = pd.DataFrame(end_search_repair,columns=[
    "dose","rate","unit_zero","rate_unit_zero","ekind_zero","rmse"])
print(end_search_table_repair.sort_values("rmse").head(12).to_string(index=False))
```
