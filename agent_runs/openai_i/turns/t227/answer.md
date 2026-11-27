Reserving the head leaves about 5.3% held-stay error in the attention output, regardless of anchor size. That is not small enough to assume factual safety. I’ll install a reversible **zero-effect** transplant and test its factual NLL before adding any drug response. If this baseline already exceeds the 0.005-nat limit, the channel is unsuitable.

```python
anchor203 = 100.
trial203 = neutral_trials201[anchor203]
gamma203 = gamma202.astype(np.float64)
beta203 = beta202.astype(np.float64)
constant203 = (1. / gamma203) / np.sum(beta203 / gamma203)
head_start203 = int(private_head196) * 48
head_slice203 = slice(head_start203, head_start203 + 48)

q203 = array196(load_original("core.prelude.0.attn.q.weight")).copy()
kv203 = array196(load_original("core.prelude.0.attn.kv.weight")).copy()
q203[head_slice203] = 0
kv203[head_slice203] = 0
kv203[192+head_start203:192+head_start203+48] = 0

q203[head_start203] = (np.sqrt(48.) * constant203).astype(np.float32)
kv203[head_start203] = (50. * anchor196 / (192. * gamma203)).astype(np.float32)
kv203[192+head_start203] = constant203.astype(np.float32)
for j, axis in enumerate((axis_ins196, axis_ne196)):
    value_row = anchor203 * axis.astype(np.float64) / (192. * gamma203)
    value_row -= float(value_row @ beta203) * constant203
    kv203[192+head_start203+1+j] = value_row.astype(np.float32)

out203 = np.zeros((192, 192), dtype=np.float32)
out203[:, keep_att201] = trial203["coef"][:-1].T.astype(np.float32)
out203[:, head_start203] = trial203["coef"][-1].astype(np.float32)

routed203 = attention_features196(
    trial203["tokens"], torch.tensor(q203), torch.tensor(kv203))
features203 = routed203[0] if isinstance(routed203, tuple) else routed203
private_values203 = array196(features203)[:, :, head_start203:head_start203+3]
expected203 = np.broadcast_to(F195[source201, None, :],
                             private_values203[:, :, 1:].shape)
print("Constant channel max error:", float(np.max(np.abs(private_values203[:, :, 0]-1))))
print("Innovation channel RMS errors:",
      np.sqrt(np.mean((private_values203[:, :, 1:]-expected203)**2, axis=(0, 1))))
reconstructed203 = array196(torch.nn.functional.linear(features203, torch.tensor(out203)))
print("Routed neutral held attention RMS:",
      float(np.sqrt(np.mean((reconstructed203.reshape(-1, 192)[~fit_att201]
                            -target_att201[~fit_att201])**2))))

neutral203 = {k: v.clone() for k, v in private_post201.items()}
neutral203.update({
    "z_in.weight": torch.tensor(trial203["zinW"], dtype=torch.float32),
    "z_in.bias": torch.tensor(trial203["zinb"], dtype=torch.float32),
    "core.prelude.0.attn.q.weight": torch.tensor(q203),
    "core.prelude.0.attn.kv.weight": torch.tensor(kv203),
    "core.prelude.0.attn.out.weight": torch.tensor(out203)
})
np.savez("/work/neutral203.npz", **{k: array196(v) for k, v in neutral203.items()})
assert not edited(), "Unexpected active edits."
for k, v in neutral203.items():
    save_edit(k, v)

factual_reference203 = pd.read_csv("/work/fwd_14/factual_original.csv")
print("Factual reference columns:", list(factual_reference203.columns))
stay_col203 = next(c for c in ("stay_id", "stay") if c in factual_reference203.columns)
validation_stays203 = sorted(factual_reference203[stay_col203].unique().tolist())
print("Validation stays:", validation_stays203)
print("Active edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 203 — install zero-effect private transport\n"
            "- Reserved head routes a constant and two gated dose innovations "
            "from the re-encoded latent token; drug-response columns remain zero.\n"
            "- Remaining attention output reconstructed by held-stay ridge regression "
            "(about 5.3% relative RMS error).\n"
            "- Saved reversible neutral203.npz and installed nine tensors solely "
            "for factual safety testing; no causal correction is active.\n")
```
