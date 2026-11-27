The edit uses two reserved latent channels, and the core replay is available separately from the unreliable head replay. I’ll measure those channels under recorded and doubled doses. Clear, drug-specific responses would shift attention to downstream routing; a weak or saturated channel would need repair first.

```python
print("=== CANDIDATE DIFFERENCES FROM ZERO-CHANNEL BASELINE ===")
for _label205, _cand205 in [("139", candidate139), ("193", candidate193),
                           ("199", candidate199)]:
    print(_label205, {k: float(torch.linalg.vector_norm(
        _cand205[k] - zero_weights154[k])) for k in zero_weights154})

def latent205(weights, u, context=context151):
    x = torch.cat([torch.as_tensor(context["ctx"], dtype=torch.float32),
                   torch.as_tensor(context["e"], dtype=torch.float32),
                   torch.as_tensor(u, dtype=torch.float32)], dim=1)
    hidden = F.gelu(F.linear(x, weights["z_post.0.weight"],
                            weights["z_post.0.bias"]))
    return F.linear(hidden, weights["z_post.2.weight"],
                    weights["z_post.2.bias"])[:, :32]

_z205 = latent205(candidate199, context151["u"]).numpy()
_zcf205 = latent205(candidate199, ucf151).numpy()
print("\n=== ACTUAL RESERVED CHANNELS ===")
print("drug labels:", np.unique(drug151, return_counts=True))
for _drug205 in np.unique(drug151):
    _m205 = np.asarray(drug151) == _drug205
    print(str(_drug205), "factual mean/std:",
          _z205[_m205, 30:32].mean(0), _z205[_m205, 30:32].std(0),
          "doubled minus recorded mean:",
          (_zcf205 - _z205)[_m205, 30:32].mean(0),
          "delta quantiles:",
          np.quantile((_zcf205 - _z205)[_m205, 30:32], [.1, .5, .9], axis=0))
print("empty-action channel mean:",
      latent205(candidate199, torch.zeros_like(context151["u"]))[:, 30:32].mean(0))

print("\n=== PROBE INDEX AND RECORD INTERFACES ===")
for _k205 in sorted(globals()):
    if _k205.startswith("rr33") or _k205 in [
        "indices151", "events151", "dose_feature_fit", "dose_feature_coef",
        "Bcomp89", "Binnov89", "r186_idx", "r186_u", "cols185"]:
        print(_k205, brief202(globals()[_k205]))
for _k205 in ["embed_consistent_audit", "encode_sets_audit",
              "encode_single_audit", "embed_record_audit"]:
    if _k205 in globals() and isinstance(globals()[_k205], types.FunctionType):
        print(_k205, args203(globals()[_k205]),
              "constants:", str(globals()[_k205].__code__.co_consts)[:350])

print("\n=== CONSTRUCTION NOTES ===")
_n205 = Path("/work/notes.md").read_text()
for _heading205 in ["## Run 132:", "## Run 139:"]:
    _a205 = _n205.find(_heading205)
    if _a205 >= 0:
        _b205 = _n205.find("\n## ", _a205 + 4)
        print(_n205[_a205:_b205 if _b205 >= 0 else None][:1000])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 205: directly measure reserved-dose channels\n"
            "- Measuring factual, doubled-dose, and empty-action posterior "
            "channels using the actual saved posterior weights, without a "
            "value-head surrogate. No weights changed.\n"
            "- Python runs: 205; forward passes: 14.\n")
```