This repeats the register-probe result; it does not yet test the factual head. I’ll run that check now with the recovered register fields. Matching the saved deterministic predictions to rounding precision would confirm the remaining computation.

```python
import torch.nn.functional as F
r91_W = {n: load_original(n) for n in NAMES}
def r91_lin(x, p):
    return F.linear(x, r91_W[p+".weight"], r91_W.get(p+".bias"))
def r91_norm(x, p):
    return F.layer_norm(x, (192,), r91_W[p+".weight"], r91_W[p+".bias"], 1e-5)
def r91_mlp(x, p, act):
    x = r91_lin(x, p+".0")
    return r91_lin(F.gelu(x) if act == "gelu" else F.silu(x), p+".2")

r91_tool = pd.read_csv("/work/fwd_8/factual_edited.csv")
assert r91_tool[r90_keys].reset_index(drop=True).equals(r90_meta[r90_keys])
r91_nf = torch.zeros(r90_n, 8)
r91_nf[:, 0] = torch.tensor(r90_has)
r91_nf[:, 3] = torch.tensor(np.log1p(r90_age)/10)
r91_nf[:, 6] = torch.tensor(r90_z)
r91_code, r91_lb = torch.tensor(r90_codeid), torch.tensor(r90_bins["left"])
r91_fixed = (r91_W["embed.code.weight"][r91_code] + r91_lin(r91_nf, "last_numeric")
    + r91_W["embed.bins.value.weight"][r91_lb]
    + r91_W["embed.bins.rate.weight"][0] + r91_W["last_status.weight"][0])
r91_tr = r91_W["embed_transition.value_bin.weight"][r91_code*17+r91_lb]
r91_centres = torch.tensor(r90_centres, dtype=torch.float32)
r91_scale = torch.tensor([r90_numeric[(ev,"value")]["iqr"] for ev in r90_event])
r91_tm = torch.tensor(r91_tool.predicted_mean.to_numpy(), dtype=torch.float32)
r91_tn = torch.tensor(r91_tool.nll.to_numpy(), dtype=torch.float32)
r91_dt = torch.tensor(r90_dt, dtype=torch.float32)
r91_obin = np.array([1+np.searchsorted(
    r90_numeric[(ev,"value")]["fine_bin_edges"], val, side="left")
    for ev,val in zip(r90_event,r90_meta.observed)])
r91_obin = torch.tensor(r91_obin)
r91_gaps = {"seconds/10":torch.log1p(r91_dt)/10,
            "logseconds/10":torch.log(r91_dt.clamp_min(1))/10,
            "seconds":torch.log1p(r91_dt),
            "hours":torch.log1p(r91_dt/3600)}
r91_results, r91_best = [], None
with torch.no_grad():
    for flow in [True,False]:
        h = R85["h"].float()
        if flow:
            h = h*torch.exp(-r91_dt[:,None,None]*torch.exp(r91_W["core.flow.a"]))
        base = R85["out"].float()+r91_norm(h.mean(1),"core.out_norm")
        for act in ["gelu","silu"]:
            for gn,gi in r91_gaps.items():
                view = base+r91_mlp(gi[:,None],"core.gap",act)
                for ti in range(15):
                    q = r91_norm(view+r91_W["type_emb.weight"][ti],"query_norm")
                    wc = q+r91_fixed
                    pre = r91_lin(wc,"attr_heads.value_bin.0")
                    hid = F.gelu(pre) if act=="gelu" else F.silu(pre)
                    logits = r91_lin(hid,"attr_heads.value_bin.2")+r91_tr
                    lp = F.log_softmax(logits,1)
                    mu = (lp.exp()*r91_centres).sum(1)
                    nll = -lp[torch.arange(r90_n),r91_obin]
                    me = (((mu-r91_tm)/r91_scale)**2).mean().sqrt().item()
                    ne = ((nll-r91_tn)**2).mean().sqrt().item()
                    cfg = (me,ne,flow,act,gn,ti)
                    r91_results.append(cfg)
                    if r91_best is None or me<r91_best["config"][0]:
                        r91_best = dict(config=cfg,q=q.clone(),with_code=wc.clone(),
                            pre=pre.clone(),hidden=hid.clone(),logits=logits.clone(),
                            mean=mu.clone(),nll=nll.clone(),view=view.clone())
print("Best (normalized mean RMSE, NLL RMSE, flow, activation, gap, type):")
for cfg in sorted(r91_results)[:8]:
    print(cfg)
print("Maximum mean error:",float((r91_best["mean"]-r91_tm).abs().max()))
print("Edits:",edited())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 91\nExact register recovered; scanning deterministic head "
            "against forward 8. No weights edited.\n")
```