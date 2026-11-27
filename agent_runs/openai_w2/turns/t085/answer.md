Single-start bundles now reproduce to about \(3\times10^{-5}\), validating the context and action encoder for those cases. END bundles still do not. I’ll test whether END atoms are routed differently or use a different kind/quality embedding. A similarly tiny error would establish the convention; otherwise I’ll restrict the analysis to verified bundles.

```python
print("Extra embedding tables:")
print([(n, SHAPES[n]) for n in NAMES if n.startswith("embed.") and
       ("categorical" in n or "kq" in n)])

ec = end_cases_repair if "end_cases_repair" in globals() else ec
ctx, _ = previous_context_repair([c[0] for c in ec])
target = sx_repair[[c[0] for c in ec],960:992]
starts = torch.stack([embed_record_atom(c[1][0]) for c in ec])
u_start = action_encode_atoms(starts[:,None,:])
zh = rmlp(torch.cat([ctx,empty_e_repair.expand(len(ec),-1),u_start],1),"z_post")[:,:32]
print("Ignore END entirely RMSE:",float(((zh-target)**2).mean().sqrt()))

def batched_set_repair(atoms, prefix):
    n = atoms.shape[1]
    pooled = rmlp(atoms, prefix+".mlp1").sum(1)/np.sqrt(n)
    pooled += rlinear(torch.full((len(atoms),1),np.log1p(n)),prefix+".size")
    return rnorm(rmlp(pooled,prefix+".mlp2"),prefix+".norm")

end_route_results_repair = []
for mode, kind, kq in itertools.product(
    ("same","absent_with_units","absent_no_units"),
    (0,1,2), range(Wrepair["embed.kq.weight"].shape[0])
):
    endings = []
    for c in ec:
        row = dict(c[2][0])
        if mode!="same":
            row["dose"],row["rate"] = np.nan,np.nan
        a = embed_record_atom(row,kind1=kind,end_mode1="same")
        if mode=="absent_no_units":
            w = Wrepair["embed.categorical.unit.weight"]
            a = a-w[unit_ids_repair[row["event"]]]+w[0]
            w = Wrepair["embed.categorical.rate_unit.weight"]
            a = a-w[rate_unit_ids_repair[row["event"]]]+w[0]
        w = Wrepair["embed.kq.weight"]
        a = a-w[0]+w[kq]
        endings.append(a)
    endings = torch.stack(endings)
    u_both = action_encode_atoms(torch.stack([starts,endings],1))
    e_end = batched_set_repair(endings[:,None,:],"encode")
    for route,e,u in (
        ("action",empty_e_repair.expand(len(ec),-1),u_both),
        ("observation",e_end,u_start),
        ("both",e_end,u_both)
    ):
        zh = rmlp(torch.cat([ctx,e,u],1),"z_post")[:,:32]
        end_route_results_repair.append(
            (mode,kind,kq,route,float(((zh-target)**2).mean().sqrt()))
        )
end_route_table_repair = pd.DataFrame(end_route_results_repair,
    columns=["mode","kind","kq","route","rmse"])
print(end_route_table_repair.sort_values("rmse").head(15).to_string(index=False))
```
