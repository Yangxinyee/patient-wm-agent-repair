The four slots have almost identical reading sensitivities, but MAP and glucose vary across state directions and contexts. I can therefore simplify the fit to slot-mean features. Next I’ll verify the reconstruction on all retained bundles and recover full value probabilities, so the fit can protect factual distributions—not just their means. Large reconstruction errors would invalidate that fit.

```python
import types, time
_t121 = time.time()
n_bundle121 = len(state_df)
hnew_all121 = torch.empty((n_bundle121, 4, 192), dtype=torch.float32)
delta121 = torch.nn.functional.softplus(W0["core.delta_in"])
abar121 = torch.exp(-delta121 * torch.exp(W0["core.a_in"]))
for _start121 in range(0, n_bundle121, 256):
    _rows121 = np.arange(_start121, min(_start121 + 256, n_bundle121))
    _mg121 = dict(make_core_memory.__globals__)
    _mg121["core_rows"] = _rows121
    _mg121["core_e_all"] = e_fact94
    _mf121 = types.FunctionType(make_core_memory.__code__, _mg121)
    _mem121, _age121, _valid121 = _mf121(True)
    _ages121 = core_ffn(torch.log1p(_age121.clamp_min(0)), "core.age")
    _h121 = hflow46[_rows121]
    _E121 = torch.as_tensor(E_fact95[_rows121], dtype=torch.float32)
    _base121 = _h121 + _E121
    _uu121 = _base121
    for _prefix121 in ("core.blocks.0.0", "core.blocks.0.1"):
        _uu121 = core_layer(_uu121, _prefix121, 4,
                           _mem121, _ages121, _valid121, "both")
    hnew_all121[_rows121] = (
        abar121 * _h121 + delta121 * core_lin(_E121, "core.b_in")
        + (_uu121 - _base121)
    )
hlocal121, olocal121 = coda46(hnew_all121)
out_error121 = (olocal121 - core_state[:, :192]).detach().numpy()
print("local state out RMSE:", np.sqrt(np.mean(out_error121 ** 2)))
print("per-bundle out RMSE quantiles:",
      np.quantile(np.sqrt(np.mean(out_error121 ** 2, axis=1)), [0, .5, .9, 1]))
print("original b_in consistency:",
      float(torch.max(torch.abs(W0["core.b_in.weight"]
                                - torch.as_tensor(load_original("core.b_in.weight"))))))

src_all121 = torch.as_tensor(head_meta.state_row.to_numpy(), dtype=torch.long)
dt_all121 = torch.as_tensor(head_meta.dt.to_numpy(), dtype=torch.float32)[:, None]
_glob121 = dict(next_means45.__globals__)
_glob121.update(src45=src_all121, dt45=dt_all121, static45=head_static,
                trans45=head_transition, centres45=head_centres)
means_all121 = types.FunctionType(next_means45.__code__, _glob121)

def readout121(h, out, src=src_all121, dt=dt_all121,
               static=head_static, transition=head_transition, centres=head_centres):
    flowed = h[src] * torch.exp(
        -dt[:, :, None] * torch.exp(W0["core.flow.a"])[None, None, :]
    )
    ctx = (out[src] + norm0(flowed.mean(dim=1), "core.out_norm")
           + mlp0(torch.log(dt.clamp_min(1e-6)) / 10.0, "gap"))
    zmean = mlp0(ctx, "z_prior")[:, :32]
    view = ctx + linear0(zmean, "z_head")
    q = norm0(view + W0["type_emb.weight"][0], "query_norm")
    hidden = torch.nn.functional.gelu(linear0(q + static, "attr_heads.value_bin.0"))
    logits = linear0(hidden, "attr_heads.value_bin.2") + transition
    prob = logits.softmax(-1)
    return (prob * centres).sum(-1), logits, prob, hidden

mean_all121, logits_all121, prob_all121, hidden_all121 = readout121(hlocal121, olocal121)
_check121 = means_all121(hlocal121, olocal121)
print("generic decoder / existing decoder max difference:",
      float(torch.max(torch.abs(mean_all121 - _check121))))
print("local mean / GPU mean RMSE:",
      np.sqrt(np.mean((mean_all121.numpy() - head_meta.gpu_mean.to_numpy()) ** 2)))
print("E contrast shapes:", {k: tuple(v.shape) for k, v in E_edit95.items()})
print("NE target shape:", np.asarray(bundle_target115(("norepinephrine", 2.0))).shape)
print("input basis orthogonality error:",
      np.max(np.abs(basis100.T @ basis100 - np.eye(basis100.shape[1]))))
print("elapsed seconds:", round(time.time() - _t121, 2))
with open("/work/notes.md", "a") as _f121:
    _f121.write("\n## Run 121: all-bundle reconstruction and distributional readout\n"
                "Slot sensitivities are nearly identical, supporting a slot-mean structured regression. "
                f"Reconstructed all {n_bundle121} bundles; out RMSE={np.sqrt(np.mean(out_error121**2)):.6f}. "
                "Implemented generic mean/logit/probability readout and checked against retained decoder. "
                "No weights changed. Python runs: 121; forwards: 15.\n")
```