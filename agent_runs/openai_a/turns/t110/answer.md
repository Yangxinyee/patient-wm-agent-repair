The omitted categorical fields have zero embeddings, so they do not explain the error. Empty-world handling and time encoding remain plausible causes. I’ll compare those conventions against the saved posterior means; near-zero error would validate reconstruction, while a substantial residual would refute it.

```python
import itertools
import torch.nn.functional as F

torch.set_num_threads(8)
_r96_B = len(rm_rates)
_r96_dt = rm_dt.float().reshape(_r96_B, 1).clamp_min(1e-6)
_r96_prev = rm_prev.float()
_r96_h = _r96_prev[:, 192:960].reshape(_r96_B, 4, 192)
_r96_out = _r96_prev[:, :192]
_r96_target = rm_target_z.float()

print("Absent-bin norms:", {
    a: float(rm_W[f"embed.bins.{a}.weight"][0].norm())
    for a in ("value", "dose", "rate")
})

def r96_empty_world(size_n, atom=None):
    pooled = torch.zeros(_r96_B, 192)
    if atom is not None:
        pooled = rm_mlp(atom.reshape(_r96_B, 192), "encode.mlp1")
    pooled += rm_lin(
        torch.full((_r96_B, 1), float(np.log1p(size_n))), "encode.size"
    )
    return rm_ln(rm_mlp(pooled, "encode.mlp2"), "encode.norm")

_r96_empty_cases = {
    "zero": torch.zeros(_r96_B, 192),
    "empty_size0": r96_empty_world(0),
    "empty_size1": r96_empty_world(1),
    "zero_atom": r96_empty_world(1, torch.zeros(_r96_B, 192)),
    "numeric_bias_atom": r96_empty_world(
        1, rm_W["embed.numeric.bias"].expand(_r96_B, -1)
    ),
}
_r96_contexts = {}
for _r96_clock in ("raw", "rounded_seconds"):
    _r96_d = (_r96_dt if _r96_clock == "raw"
             else _r96_dt.round().clamp_min(1))
    _r96_hf = _r96_h * torch.exp(
        -_r96_d[:, :, None] * rm_W["core.flow.a"].exp()[None, None, :]
    )
    for _r96_flow in ("flowed", "unflowed"):
        _r96_hc = _r96_hf if _r96_flow == "flowed" else _r96_h
        _r96_base = _r96_out + rm_ln(
            _r96_hc.mean(1), "core.out_norm"
        )
        for _r96_gap in ("log_div10", "log1p_div10", "log_raw"):
            _r96_t = (
                _r96_d.log() / 10 if _r96_gap == "log_div10"
                else _r96_d.log1p() / 10 if _r96_gap == "log1p_div10"
                else _r96_d.log()
            )
            _r96_contexts[(_r96_clock, _r96_flow, _r96_gap)] = (
                _r96_base + rm_mlp(_r96_t, "gap")
            )

r96_results = []
r96_best = None
for _r96_unit, _r96_rateunit, _r96_side, _r96_kq in itertools.product(
    range(6), range(4), ("left", "right"), (False, True)
):
    _r96_atoms = rm_insulin_atoms(
        _r96_unit, _r96_rateunit, _r96_side, _r96_kq, "gelu"
    )
    _r96_u = rm_set(_r96_atoms, "encode_action")
    _r96_ecases = dict(_r96_empty_cases)
    _r96_ecases["drug_in_world"] = rm_set(_r96_atoms, "encode")
    for _r96_ck, _r96_ctx in _r96_contexts.items():
        for _r96_ek, _r96_e in _r96_ecases.items():
            _r96_mu = rm_mlp(
                torch.cat([_r96_ctx, _r96_e, _r96_u], 1), "z_post"
            )[:, :32]
            _r96_err = float((_r96_mu - _r96_target).square().mean())
            _r96_spec = (
                _r96_unit, _r96_rateunit, _r96_side, _r96_kq,
                _r96_ck, _r96_ek
            )
            r96_results.append((_r96_err, _r96_spec))
            if r96_best is None or _r96_err < r96_best["mse"]:
                r96_best = {
                    "mse": _r96_err, "spec": _r96_spec,
                    "mu": _r96_mu.clone(), "ctx": _r96_ctx.clone(),
                    "e": _r96_e.clone(), "u": _r96_u.clone(),
                    "atoms": _r96_atoms.clone()
                }

print("Best posterior reconstructions (RMSE):")
for _r96_err, _r96_spec in sorted(r96_results)[:12]:
    print(round(_r96_err ** 0.5, 8), _r96_spec)
print("Target SD:", float(_r96_target.std()))
print("Best correlation:", np.corrcoef(
    r96_best["mu"].numpy().ravel(), _r96_target.numpy().ravel()
)[0, 1])

with Path("/work/notes.md").open("a") as _r96_f:
    _r96_f.write(
        "\n## Empty-world/time reconstruction search; Python 96\n"
        f"- Best posterior reconstruction: {r96_best['spec']}; "
        f"RMSE={r96_best['mse'] ** .5:.8g}.\n"
        "- Absent categorical embeddings are zero. No weight edits "
        "were made by this reconstruction experiment.\n"
    )
```
