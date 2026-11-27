Bin boundaries and bin-centre registers do not explain the mismatch. Four-draw sampling accounts for some, but not all, of it. I’ll make one bounded check of query types and context embeddings. A materially smaller error would identify the missing convention; otherwise I’ll use the GPU tool, not the surrogate, as the decisive check.

```python
# Bounded search over discrete conventions; no weight changes.
ctx_base = context_base
gap_default = mlp((torch.log1p(dt)/10)[:,None], "gap", "gelu")
ctx_default = ctx_base + gap_default
z_default = mlp(ctx_default, "z_prior", "gelu")[:,:32]
view_default = ctx_default + TF.linear(z_default, wt("z_head.weight"), wt("z_head.bias"))
reg_default = register.clone()

variant_results = []
variant_features = {}
def score_head_variant(label, query_input, reg=reg_default, normalize=True):
    query = ln(query_input, "query_norm") if normalize else query_input
    inputs = query + reg
    hidden = TF.gelu(TF.linear(inputs, wt("attr_heads.value_bin.0.weight"),
                               wt("attr_heads.value_bin.0.bias")))
    lp = TF.linear(hidden, wt("attr_heads.value_bin.2.weight"),
                   wt("attr_heads.value_bin.2.bias")) + transition_t
    pp = lp.softmax(-1)
    means = (pp*centres_t).sum(-1).numpy()
    nll = -pp[torch.arange(len(hm)),target_t].clamp_min(1e-12).log().numpy()
    variant_results.append({
        "variant":label, "scaled_mean_rmse":np.sqrt(np.mean(((means-target_mu)/scale_mu)**2)),
        "nll_rmse":np.sqrt(np.mean((nll-hm.original_nll.to_numpy())**2)),
        "mean_nll":nll.mean()
    })
    variant_features[label] = inputs.numpy()

for type_i in range(15):
    score_head_variant(f"type_{type_i}", view_default + wt("type_emb.weight")[type_i])

for seconds_unit in [1/3600, 1/60, 60, 1000]:
    gg = mlp((torch.log1p(dt*seconds_unit)/10)[:,None], "gap", "gelu")
    cc = ctx_base + gg
    zm = mlp(cc, "z_prior", "gelu")[:,:32]
    vv = cc + TF.linear(zm, wt("z_head.weight"), wt("z_head.bias"))
    score_head_variant(f"gap_unit_{seconds_unit:g}", vv + wt("type_emb.weight")[1])

for name in ["embed.kq.weight","embed.kind.weight","embed.ekind.weight"]:
    for ii in range(SHAPES[name][0]):
        score_head_variant(f"query_plus_{name}_{ii}",
                           view_default + wt("type_emb.weight")[1] + wt(name)[ii])
        score_head_variant(f"register_plus_{name}_{ii}",
                           view_default + wt("type_emb.weight")[1],
                           reg_default + wt(name)[ii])

flowed_summary = ln(hflow.mean(1), "core.out_norm")
for label, base in [
    ("double_flowed_summary",2*flowed_summary),
    ("out_only",out),
    ("flowed_summary_only",flowed_summary),
    ("summary_plus_slots",out + ln((hflow+wt("core.slots")[None,:,:]).mean(1),"core.out_norm"))
]:
    cc = base + gap_default
    zm = mlp(cc,"z_prior","gelu")[:,:32]
    vv = cc + TF.linear(zm,wt("z_head.weight"),wt("z_head.bias"))
    score_head_variant(label, vv + wt("type_emb.weight")[1])

score_head_variant("without_query_norm",view_default + wt("type_emb.weight")[1], normalize=False)
variant_table = pd.DataFrame(variant_results).sort_values("scaled_mean_rmse")
print(variant_table.head(16).to_string(index=False))

print("\nState statistics:")
print("out sd:", float(out.std()), "h sd:",float(hh.std()),
      "mean slot sd:",float(hh.mean(1).std()))
print("\nKalman objective argument names:", kalman_nll.__code__.co_varnames[:22])
print("Kalman objective numeric constants:",
      [v for v in kalman_nll.__code__.co_consts if isinstance(v,(int,float,tuple))][:25])

with open("/work/notes.md","a") as f:
    f.write("\n## Reconstruction diagnostics\n"
            "- Best initial conventions: GELU, type 1, interleaved last numeric, status 0, seconds/10.\n"
            "- Reconstruction scaled mean RMSE ~0.115; expected four-prior-draw noise ~0.0475.\n"
            "- Bin boundary, centre-value, and latent averaging checks did not explain the remaining error.\n"
            "- Performing a bounded query/context convention check; no edits.\n"
            "- Python runs: 19. Forward passes: 4.\n")
```
