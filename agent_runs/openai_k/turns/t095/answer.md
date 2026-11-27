The innovation decoder responds nearly linearly to changed doses, and compressing the original token to 30 coordinates loses very little. The state sensitivities vary substantially, however. I’ll fit each response direction only on bundles containing its drug, then test the combined edit locally. It must improve dose responses without exceeding the factual-loss budget before I submit it for an external test.

```python
t89 = time.time()
Binnov89 = np.asarray(post66_rate_models[1.0], dtype=np.float64).copy()
Binnov89[:,1] = np.asarray(post66_rate_models[0.1])[:,1] / 0.8329
print("conditional token sensitivity norms:")
for d89 in range(2):
    active89 = rate66[probe88,d89] > 0
    print(d89, int(active89.sum()),
          np.round(np.linalg.norm(J88[active89].mean(0),axis=1),4))

rng89 = np.random.default_rng(189)
CF89 = []
indices66_89 = np.asarray(index66,dtype=int)
for d89 in range(2):
    select89 = ((np.arange(len(index66)) >= 4035)
                & (rate66[indices66_89,d89] > 0)
                & np.isclose(dr66[:,d89],rate66[indices66_89,d89],atol=1e-5)
                & np.isclose(dr66[:,1-d89],0,atol=1e-6)
                & (lookup80[indices66_89] >= 0))
    available89 = np.flatnonzero(select89)
    chosen89 = rng89.choice(available89, min(160,len(available89)), replace=False)
    CF89.append({"drug":d89, "indices":indices66_89[chosen89],
                 "u":torch.tensor(np.asarray(X66)[chosen89,-192:],dtype=torch.float32),
                 "dc":dr66[chosen89,d89]})

base_mlp89 = mlp0
def hook_mlp89(x,prefix,kind="gelu"):
    if prefix == "z_prior":
        capture74.append(x.detach().clone())
    return base_mlp89(x,prefix,kind)
mlp0 = hook_mlp89
baseline_slopes89 = []
try:
    for c89 in CF89:
        _,cm89,_ = dataset80(c89["indices"],u=c89["u"])
        bm89 = M0_80[lookup80[c89["indices"]]]
        delta89 = cm89-bm89
        slopes89 = (delta89*c89["dc"][:,None]).sum(0)/(c89["dc"]**2).sum()
        baseline_slopes89.append(slopes89)
    print("original local x2 slopes:",np.round(baseline_slopes89,3))
finally:
    mlp0 = base_mlp89

D89 = np.zeros((192,2))
for d89 in range(2):
    act89 = rate66[probe88,d89] > 0
    aj89 = J88[act89].reshape(-1,192)
    tj89 = np.tile(target88[:,d89]-baseline_slopes89[d89], int(act89.sum()))
    D89[:,d89] = np.linalg.solve(aj89.T@aj89/len(aj89)+1e-3*np.eye(192),
                                aj89.T@tj89/len(aj89))
    print("direction",d89,"norm",round(float(np.linalg.norm(D89[:,d89])),3),
          "conditional slopes",np.round(J88[act89]@D89[:,d89],3).mean(0))

Bp89 = np.vstack([arr71(post66_base_weights["z_post.2.weight"][:32]).T,
                   arr71(post66_base_weights["z_post.2.bias"][:32])])
Btoken89 = Bp89@zinW88.T
Btoken89[-1] += zinb88
Bcomp89 = Btoken89@V88
Bcomp89[-1] -= token_mu88@V88

def build_candidate89(scale):
    w89 = {k:v.detach().clone() for k,v in post66_base_weights.items()}
    pw89 = w89["z_post.2.weight"].clone()
    pb89 = w89["z_post.2.bias"].clone()
    pw89[:30] = torch.tensor(Bcomp89[:-1].T,dtype=torch.float32)
    pb89[:30] = torch.tensor(Bcomp89[-1],dtype=torch.float32)
    pw89[30:32] = torch.tensor(Binnov89[:-1].T,dtype=torch.float32)
    pb89[30:32] = torch.tensor(Binnov89[-1],dtype=torch.float32)
    w89["z_post.2.weight"],w89["z_post.2.bias"] = pw89,pb89
    zi89 = np.zeros((192,32),dtype=np.float32)
    zi89[:,:30] = V88
    zi89[:,30:32] = scale*D89
    w89["z_in.weight"] = torch.tensor(zi89)
    w89["z_in.bias"] = torch.tensor(token_mu88,dtype=torch.float32)
    return w89

keys89 = list(build_candidate89(0))
saved89 = {k:W0[k].clone() for k in keys89}
origp89 = np.take_along_axis(probs0_80[keep80],bins80[:,:,None],axis=2)[:,:,0][mask80]
diagnostics89 = []
candidates89 = {}
mlp0 = hook_mlp89
try:
    for scale89 in [0.0,0.25,0.5,1.0,2.0]:
        weights89 = build_candidate89(scale89)
        W0.update(weights89)
        pf89,mf89,pr89 = dataset80(idx80)
        factp89 = np.take_along_axis(pr89[keep80],bins80[:,:,None],axis=2)[:,:,0][mask80]
        dnll89 = float(np.mean(np.log(np.maximum(origp89,1e-30))-np.log(np.maximum(factp89,1e-30))))
        slopes89 = []
        for c89 in CF89:
            _,cm89,_ = dataset80(c89["indices"],u=c89["u"])
            delta89 = cm89-mf89[lookup80[c89["indices"]]]
            slopes89.append((delta89*c89["dc"][:,None]).sum(0)/(c89["dc"]**2).sum())
        diag89 = {"scale":scale89,"dNLL":dnll89,
                  "factual_mean_RMS":np.sqrt(np.mean((mf89-M0_80)**2,axis=0)).round(4).tolist(),
                  "x2_slopes":np.round(slopes89,3).tolist()}
        diagnostics89.append(diag89)
        candidates89[scale89] = weights89
        print(diag89)
finally:
    W0.update(saved89)
    mlp0 = base_mlp89
print("seconds:",round(time.time()-t89,1),"submitted edits:",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 89: explicit innovation-to-state candidate\n"
            "- Compressed the original z_in token into 30 coordinates and reserved two posterior means for dose innovations.\n"
            "- Fitted drug-specific 192-dimensional response directions analytically; tested scale frontier locally.\n"
            "- Diagnostics in diagnostics89; candidates89 retained. Scratch weights restored; no submitted edits.\n")
```
