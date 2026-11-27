Disabling the four added gate neurons removes the no-action signal to numerical precision. I also found a channel that equals one on action bundles and zero when there is no action. I’ll use it to close the gates only on no-action bundles, then install the carrier-only decoder. The check is whether drug-bundle signals stay unchanged while no-action leakage disappears.

```python
candidate206 = {k: np.asarray(v).copy() for k,v in restricted_candidate203.items()}
_shifts206 = np.array([1024., 1280., 0., 0.], dtype=np.float64)
_w206 = candidate206["z_post.0.weight"].astype(np.float64)
_b206 = candidate206["z_post.0.bias"].astype(np.float64)
_w206[rows154,384:576] += _shifts206[:,None] * action_constant205[None,:]
_b206[rows154] -= _shifts206
candidate206["z_post.0.weight"] = _w206.astype(np.float32)
candidate206["z_post.0.bias"] = _b206.astype(np.float32)
post206 = {k: candidate206[k] for k in post187}

def latent206(u):
    z = posterior153(C154, E154, u, post206)
    raw = F.linear(z, tensor153(zin154), tensor153(zb154))
    return arr183(layer_norm(raw, "core.prelude.0.norm1"))

_zero_latent206 = latent206(np.zeros_like(U154))
_fact_latent206 = latent206(U154)
_zero_signal206 = _zero_latent206 @ restricted199
_fact_signal206 = _fact_latent206 @ restricted199
print("no-action signal RMS:", np.sqrt(np.mean(_zero_signal206**2,axis=0)))
print("factual signal change RMS:",
      np.sqrt(np.mean((_fact_signal206-_sig203)**2,axis=0)))
print("factual normalized-token RMS change:",
      np.sqrt(np.mean((_fact_latent206-xlatent199)**2)))
tokens206 = tokens192.clone()
tokens206[:,6,:] = F.linear(posterior153(C154,E154,U154,post206),
                           tensor153(zin154), tensor153(zb154))
print("factual E RMS change versus restricted candidate:",
      float(torch.sqrt(torch.mean(
          (prelude155(tokens206,candidate206)[0]-_restricted_replay203[0])**2))))

installed206 = bool(
    np.max(np.sqrt(np.mean(_zero_signal206**2,axis=0))) < 1e-4 and
    np.max(np.sqrt(np.mean((_fact_signal206-_sig203)**2,axis=0))) < .01)
if installed206:
    for _name206,_tensor206 in candidate206.items():
        save_edit(_name206, np.asarray(_tensor206,dtype=np.float32))
print("candidate206 installed:", installed206)
print("edited tensors:", edited())
with open("/work/notes.md","a") as _notes206:
    _notes206.write(
        "\nPython runs 200-206; forwards 12. Recovered pilot and exact posterior. "
        "Both full and restricted decoders transported a huge insulin signal on U=0 "
        "(about -168 to -170 dose units); attention did NOT suppress it. "
        "Disabling the four added gate neurons removes this to <1e-6. "
        "Action-normalization constant is 1 within 4e-6 on all recorded drug bundles, "
        "0 when U=0. Shifted two insulin gate biases into this action-only constant "
        "and replaced full latent decoder by two-carrier restricted199. "
        f"candidate206 installed={installed206}; reversible checkpoint193 retained. "
        "Next: authoritative factual and common-random-number rollout tests.\n")
```
