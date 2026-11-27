Latent averaging did not remove the discrepancy, so it is not the main explanation. I’ll now test the ordering of the eight previous-reading inputs and the value-head activation. A much better match would identify an encoding error without changing any weights.

```python
def act2(t, kind):
    return {"silu": F.silu, "gelu": F.gelu, "relu": F.relu, "tanh": torch.tanh}[kind](t)

context = basic + mlp(dt.log()[:, None] / 10, "gap", "silu")
zp = mlp(context, "z_prior", "gelu")
view = context + lin(zp[:, :32], "z_head")
q0 = norm(view + W["type_emb.weight"][1], "query_norm")
pre0 = lin(q0 + fixed + W["last_numeric.bias"] + W["last_status.weight"][0],
           "attr_heads.value_bin.0")
eff = W["attr_heads.value_bin.0.weight"] @ W["last_numeric.weight"]
av = torch.log1p(age) / 10
register_search = []
for zcol in range(8):
    for agecol in range(8):
        if zcol == agecol:
            continue
        rest = [j for j in range(8) if j not in [zcol, agecol]]
        flags = [(j,) for j in rest] + list(itertools.combinations(rest, 2))
        for flagcols in flags:
            pp0 = (pre0 + zv[:, None]*eff[:, zcol]
                   + av[:, None]*eff[:, agecol]
                   + has[:, None]*eff[:, list(flagcols)].sum(1))
            for ha in ["silu", "gelu", "relu"]:
                lg = lin(act2(pp0, ha), "attr_heads.value_bin.2") + transition
                p = lg.softmax(-1)
                mm = (p*ct).sum(-1)
                nn = -p[torch.arange(len(ix)), tb].log()
                register_search.append(((zcol, agecol, flagcols, ha),
                                        float(((mm-refm)**2).mean().sqrt()),
                                        float(((nn-refn)**2).mean().sqrt()), float(nn.mean())))
register_search.sort(key=lambda r: r[1])
print("Best register/activation encodings:")
for r in register_search[:12]:
    print(r)

print("\nView alternatives with the best register:")
zcol, agecol, flagcols, ha = register_search[0][0]
num_best = torch.zeros((len(ix), 8))
num_best[:, zcol] = zv
num_best[:, agecol] = av
for j in flagcols:
    num_best[:, j] = has
reg_best = fixed + lin(num_best, "last_numeric") + W["last_status.weight"][0]
view_search = []
for ga, pa, basekind, typ in itertools.product(
        ["silu", "gelu", "relu", "tanh"], ["silu", "gelu", "relu"],
        ["flow_sum", "out_only", "flow_only"], [0, 1]):
    bc = basic if basekind == "flow_sum" else hs[:, :192] if basekind == "out_only" else basic-hs[:, :192]
    cc = bc + lin(act2(lin(dt.log()[:, None]/10, "gap.0"), ga), "gap.2")
    zp = lin(act2(lin(cc, "z_prior.0"), pa), "z_prior.2")
    qq = norm(cc + lin(zp[:, :32], "z_head") + W["type_emb.weight"][typ], "query_norm")
    pp = (lin(act2(lin(qq+reg_best, "attr_heads.value_bin.0"), ha),
              "attr_heads.value_bin.2") + transition).softmax(-1)
    mm = (pp*ct).sum(-1)
    nn = -pp[torch.arange(len(ix)), tb].log()
    view_search.append(((ga,pa,basekind,typ), float(((mm-refm)**2).mean().sqrt()),
                        float(((nn-refn)**2).mean().sqrt()), float(nn.mean())))
view_search.sort(key=lambda r:r[1])
print("Best view configurations:")
for r in view_search[:8]:
    print(r)
```
