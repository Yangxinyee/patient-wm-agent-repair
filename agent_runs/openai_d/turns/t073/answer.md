The cached “original” reconstruction is no longer accurate, especially for glucose. A stale or modified weight cache could explain this without implicating the architecture. I’ll compare it with freshly loaded originals and rerun the head; finding altered cached tensors would confirm that explanation.

```python
print("=== CACHED-WEIGHT AUDIT ===")
rep_W = {n: torch.as_tensor(load_original(n), dtype=torch.float32).clone()
         for n in NAMES}
bad65 = []
for n65 in NAMES:
    if n65 in W0:
        old65 = torch.as_tensor(W0[n65], dtype=torch.float32)
        dif65 = float((old65 - rep_W[n65]).abs().max())
        if dif65 > 1e-7:
            bad65.append((n65, dif65))
print("Cached tensors differing from originals:", bad65)

def rep_linear(x, name):
    return torch.nn.functional.linear(
        x, rep_W[name + ".weight"], rep_W[name + ".bias"])

def rep_norm(x, name):
    return torch.nn.functional.layer_norm(
        x, (192,), rep_W[name + ".weight"], rep_W[name + ".bias"], 1e-5)

def rep_mlp(x, name):
    return rep_linear(torch.nn.functional.gelu(rep_linear(x, name + ".0")),
                      name + ".2")

print("\n=== REGISTER AND TIME SAMPLE ===")
print(wr_meta[["hour", "dt", "has_last", "age", "last_z", "last_bin"]]
      .head(4).to_string(index=False))
print("Cached numeric register:", wr_nf[:4].tolist())
print("Latent input versus cached prior max error:",
      float((rep_mlp(wr_base, "z_prior") - wr_prior).abs().max()))

rep_fixed65 = (
    rep_W["embed.code.weight"][wr_codes]
    + rep_W["embed.bins.value.weight"][wr_lb]
    + rep_W["embed.bins.rate.weight"][0]
    + rep_W["last_status.weight"][0]
    + rep_linear(wr_nf, "last_numeric")
)
rep_trans65 = rep_W["embed_transition.value_bin.weight"][wr_codes * 17 + wr_lb]
rep_rng65 = torch.Generator().manual_seed(6501)
rep_eps65 = torch.randn(16, len(wr_meta), 32, generator=rep_rng65)

print("\n=== FRESH ORIGINAL HEAD, CACHED CONTEXT ===")
for typ65 in [1, 6]:
    pp65 = torch.zeros(len(wr_meta), 17)
    for ee65 in rep_eps65:
        zz65 = wr_mu + torch.exp(wr_logsd) * ee65
        qq65 = rep_norm(wr_base + rep_linear(zz65, "z_head")
                        + rep_W["type_emb.weight"][typ65], "query_norm")
        hh65 = torch.nn.functional.gelu(
            rep_linear(qq65 + rep_fixed65, "attr_heads.value_bin.0"))
        ll65 = rep_linear(hh65, "attr_heads.value_bin.2") + rep_trans65
        pp65 += torch.softmax(ll65, dim=-1) / len(rep_eps65)
    mm65 = (pp65 * wr_centres).sum(-1).numpy()
    print("type", typ65, {
        ev: round(float(np.sqrt(np.mean(
            (mm65[wr_meta.event.to_numpy() == ev] -
             wr_meta.tool_mean.to_numpy()[wr_meta.event.to_numpy() == ev])**2))), 4)
        for ev in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
    })

print("\nFresh original head tensor norms:", {
    n: round(float(rep_W[n].norm()), 4)
    for n in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.2.weight"]
})
print("Current edits:", edited())
```