The fitted register mapping remained inaccurate and had no simple structure, weakening the register-only explanation. I’ll now test how the recurrent state and latent enter the prediction head, with weights unchanged. A clear improvement from one convention would identify a deeper reconstruction error.

```python
print("=== STATE / LATENT READOUT CONVENTION ASSAY ===")
out120 = x104[:,:192]
flow120 = norm0(flow104.mean(1),"core.out_norm")
gap120 = mlp0(torch.log(dt104)[:,None]/10,"gap")
latent120 = {
    "prior_samples":zh104,
    "prior_mean":lin0(mu104,"z_head")[:,None,:],
    "previous_posterior":lin0(x104[:,960:992],"z_head")[:,None,:],
    "zero":torch.zeros((len(ix104),1,192)),
}
attrs120 = {}
for vc120,fc120 in ((6,3),(2,3),(6,7)):
    numeric120 = torch.zeros((len(ix104),8))
    numeric120[:,0] = has118
    numeric120[:,1] = age118
    numeric120[:,vc120] = z118
    numeric120[:,fc120] = has118
    attrs120[(vc120,fc120)] = (
        codeemb104 + W0["embed.bins.value.weight"][bins118]
        + W0["embed.bins.rate.weight"][0]
        + W0["last_status.weight"][0]
        + lin0(numeric120,"last_numeric"))
results120 = []
for ao120,af120 in ((1,1),(1,0),(0,1),(2,1),(1,2),(.5,.5),(2,0)):
    for gs120 in (1,10):
        for lname120,l120 in latent120.items():
            view120 = (ao120*out120 + af120*flow120 + gs120*gap120)[:,None,:] + l120
            q120 = norm0(view120+W0["type_emb.weight"][1],"query_norm")
            for layout120,add120 in attrs120.items():
                pre120 = F.linear(q120+add120[:,None,:],
                                  W0["attr_heads.value_bin.0.weight"],
                                  W0["attr_heads.value_bin.0.bias"])
                for act120 in ("gelu","relu"):
                    logits120 = F.linear(act0(pre120,act120),
                                         W0["attr_heads.value_bin.2.weight"],
                                         W0["attr_heads.value_bin.2.bias"]) + tr118[:,None,:]
                    p120 = torch.softmax(logits120,-1).mean(1)
                    means120 = (p120*centres104).sum(-1).numpy()
                    nlls120 = -torch.log(p120[torch.arange(len(ix104)),labels118]
                                        .clamp_min(1e-12)).numpy()
                    results120.append({
                        "out":ao120,"flow":af120,"gap":gs120/10,
                        "latent":lname120,"layout":layout120,"activation":act120,
                        "objective":float(np.mean(np.abs(means120-target_mean118)/iqr118)
                                          +np.mean(np.abs(nlls120-target_nll118))),
                        "mean_mae":float(np.mean(np.abs(means120-target_mean118))),
                        "nll_mae":float(np.mean(np.abs(nlls120-target_nll118)))})
results120.sort(key=lambda d:d["objective"])
for d120 in results120[:10]:
    print(d120)
print("Prior ranges:",float(mu104.min()),float(mu104.max()),
      float(ls104.min()),float(ls104.max()))
print("Edits:",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 120: state/latent readout convention assay\n"
            "- Register-only reconstruction is inadequate; tested state, flow, gap, and latent conventions with original weights.\n"
            "- Best result: "+str(results120[0])+"\n"
            "- No edits. Python runs: 120; forward passes: 4.\n")
```