The reconstruction favors measurement type 1, interleaved numeric fields, and status 0, but its errors are still too large for a precise edit. I’ll test whether latent-prior averaging and timing conventions explain the discrepancy. If they do not, I won’t treat this local head reconstruction as exact.

```python
W_original = {}
def wt(name):
    if name not in W_original:
        W_original[name] = torch.as_tensor(np.asarray(load_original(name)), dtype=torch.float32)
    return W_original[name]

print("=== ZERO-BIN NORMS ===")
for name in ["embed.bins.value.weight","embed.bins.rate.weight","embed.bins.dose.weight"]:
    print(name, float(wt(name)[0].norm()))

nums = torch.zeros((len(hm),8))
nums[:,0], nums[:,1], nums[:,2], nums[:,3] = last_has, last_age, last_z, last_has
register = (base_code + TF.linear(nums, wt("last_numeric.weight"), wt("last_numeric.bias"))
            + wt("last_status.weight")[0])
eps_mc = torch.tensor(np.random.default_rng(913).normal(size=(64,len(hm),32)), dtype=torch.float32)

integration_results = []
for act, gap_scale, flow_mode, z_mode in product(
        ["gelu","silu"], [10.,1.], ["flowed","unflowed"], ["mean","mc64","posterior"]):
    flowed = hflow if flow_mode == "flowed" else hh
    gap_vec = mlp((torch.log1p(dt)/gap_scale)[:,None], "gap", act)
    ctx = out + ln(flowed.mean(1), "core.out_norm") + gap_vec
    prior_parameters = mlp(ctx, "z_prior", act)
    mean_z = prior_parameters[:,:32]
    logsd_z = prior_parameters[:,32:].clamp(-5,2)
    if z_mode == "mean":
        zz = mean_z[None,:,:]
    elif z_mode == "posterior":
        zz = sx[:,960:992][None,:,:]
    else:
        zz = mean_z[None,:,:] + logsd_z.exp()[None,:,:]*eps_mc
    views = ctx[None,:,:] + TF.linear(zz, wt("z_head.weight"), wt("z_head.bias"))
    queries = ln(views + wt("type_emb.weight")[1], "query_norm")
    xh = queries + register[None,:,:]
    hidden = activation(TF.linear(xh, wt("attr_heads.value_bin.0.weight"),
                                  wt("attr_heads.value_bin.0.bias")), act)
    logits = TF.linear(hidden, wt("attr_heads.value_bin.2.weight"),
                       wt("attr_heads.value_bin.2.bias")) + transition_t[None,:,:]
    probs = logits.softmax(-1).mean(0)
    mu = (probs*centres_t).sum(-1).numpy()
    nll = -probs[torch.arange(len(hm)),target_t].clamp_min(1e-12).log().numpy()
    integration_results.append({
        "activation":act, "gap_divisor":gap_scale, "flow":flow_mode, "z":z_mode,
        "scaled_mean_rmse":np.sqrt(np.mean(((mu-target_mu)/scale_mu)**2)),
        "nll_rmse":np.sqrt(np.mean((nll-hm.original_nll.to_numpy())**2)),
        "nll":nll.mean()
    })
integration_table = pd.DataFrame(integration_results).sort_values("scaled_mean_rmse")
print("\n=== LATENT / TIMING TEST ===")
print(integration_table.head(14).to_string(index=False))
print("\n=== PRIOR PARAMETERS, GELU / SECONDS DIVIDED BY 10 ===")
ctx = context_base + mlp((torch.log1p(dt)/10)[:,None], "gap", "gelu")
zp = mlp(ctx, "z_prior", "gelu").numpy()
print("mean quantiles:", np.quantile(zp[:,:32], [.01,.5,.99]))
print("log-sd quantiles:", np.quantile(zp[:,32:], [.01,.5,.99]))

print("\n=== RETAINED ANALYSIS FUNCTIONS ===")
import inspect
for key, obj in list(globals().items()):
    if inspect.isfunction(obj) and obj.__module__ == "__main__" and key not in {
        "wt","ln","activation","mlp","load","load_original","save_edit","drop_edit","edited"
    }:
        print(key, inspect.signature(obj))
print("\n=== KALMAN FITS ===")
for key, obj in kalman_fits.items():
    print(key, repr(obj)[:850])

print("\n=== KERNEL-RELATED OBJECT NAMES ===")
for key, obj in list(globals().items()):
    if any(s in key.lower() for s in ["kernel","bootstrap","moment","policy"]):
        print(key, type(obj).__name__, getattr(obj,"shape",""))
```
