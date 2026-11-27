The private innovations are effectively zero without actions. Transported signals are usually tiny there, but a few outliers remain, confirming some attention leakage. I’ll inspect those outliers and the bin probabilities: fitting only predicted means may also have produced directions that distort the distribution unnecessarily.

```python
with torch.no_grad():
    tn238 = torch.as_tensor(array196(tokens222), dtype=torch.float32)
    xn238 = F.layer_norm(
        tn238, (192,),
        torch.as_tensor(load_original("core.prelude.0.norm1.weight")),
        torch.as_tensor(load_original("core.prelude.0.norm1.bias")))
    qw238 = torch.as_tensor(route228["core.prelude.0.attn.q.weight"], dtype=torch.float32)
    kvw238 = torch.as_tensor(route228["core.prelude.0.attn.kv.weight"], dtype=torch.float32)
    qq238 = F.linear(xn238, qw238).reshape(len(tn238), 7, 4, 48).transpose(1, 2)
    kk238, vv238 = F.linear(xn238, kvw238).split(192, dim=-1)
    kk238 = kk238.reshape(len(tn238), 7, 4, 48).transpose(1, 2)
    pp238 = torch.softmax(qq238 @ kk238.transpose(-1, -2) / np.sqrt(48.), dim=-1)
    latent_p238 = pp238[:, 1, :4, 6].numpy()
    latent_v238 = vv238[:, 6, response_cols228].numpy()
    latent_contribution238 = latent_p238[..., None] * latent_v238[:, None, :]
    other_contribution238 = signal237 - latent_contribution238

print("=== LEAKAGE DECOMPOSITION ===")
for label, mask in (("inactive", inactive237[src237]), ("active", ~inactive237[src237])):
    print(label, "latent attention quantiles",
          np.quantile(latent_p238[mask], [0, .001, .01, .5, 1]))
    for tag, values in (("latent value", latent_v238),
                        ("other-token contribution", other_contribution238)):
        a = values[mask].reshape(-1, 2)
        print(tag, "mean", a.mean(0), "RMS", np.sqrt(np.mean(a*a, axis=0)),
              "max_abs", np.abs(a).max(0))

print("\n=== RESERVED HEAD WEIGHTS ===")
print("q row norms", np.linalg.norm(np.asarray(qw238)[48:96], axis=1)[:8])
print("neutral output-column norms",
      np.linalg.norm(route228["core.prelude.0.attn.out.weight"][:, response_cols228], axis=0))
print("value-projection norms",
      np.linalg.norm(np.asarray(kvw238)[192 + response_cols228], axis=1))

print("\n=== READOUT CONTRACTS ===")
for name in ("readout153", "readout214", "readout_capture156"):
    fn = globals()[name]
    print(name, inspect.signature(fn),
          "references", fn.__code__.co_names,
          "constants", fn.__code__.co_consts[:16])
for j, a in enumerate(base_means228):
    aa = array196(a)
    print("base_means228 component", j, aa.shape)
    if aa.ndim == 3 and aa.shape[-1] == 17:
        print("logsumexp", torch.logsumexp(torch.as_tensor(aa), dim=-1).mean(0).numpy())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 238 — separate leakage and distribution distortion\n"
            "- Private innovations on inactive bundles have RMS below 7e-15.\n"
            "- Attention signal inactive RMS about 1e-4, with rare outliers near .005; active transport slopes about .815 insulin and .831 NE.\n"
            "- Inspecting whether leakage comes from latent projection or other tokens, and whether bin-level response fitting can avoid mean-only direction damage.\n")
```