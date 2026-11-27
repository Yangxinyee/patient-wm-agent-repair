The private coordinates track dose changes approximately linearly through x0–x3, but the fitted token directions fell well short of the desired responses. Many contexts barely respond to the latent token, so the local regression is not enough. Next I’ll test finite dose changes, including a mean-response calibration. Saturation, cross-drug effects, or large factual shifts would count against it.

```python
import time
start216 = time.time()
W = Wcal214

# Minimum-norm regressions for mean primary response and mean cross-drug neutrality.
mean_directions216 = {}
for drug216 in causal_targets215:
    selected216 = (drug_labels215 == drug216) & ~hold215
    primary216 = 0 if drug216 == "insulin" else 2
    cross_event216 = 2 if drug216 == "insulin" else 0
    jmean216 = J215[selected216].mean(0)
    native_mean216 = native215[drug216][selected216].mean(0)
    AA216 = jmean216[[primary216, cross_event216]]
    yy216 = (
        causal_targets215[drug216]-native_mean216
    )[[primary216, cross_event216]]
    vv216 = np.linalg.lstsq(AA216, yy216, rcond=1e-8)[0]
    mean_directions216[drug216] = vv216
    pred216 = native215[drug216] + np.einsum("ned,d->ne", J215, vv216)
    print("MEAN-CONSTRAINED DIRECTION", drug216,
          "norm", np.linalg.norm(vv216),
          "train", pred216[selected216].mean(0),
          "held", pred216[(drug_labels215 == drug216) & hold215].mean(0))

original_z216 = mlp136(Xbase215, "z_post")[:, :32]
original_token216 = linear136(original_z216, "z_in")
neutral_w216 = torch.as_tensor(np.asarray(load("z_in.weight"))).float()
neutral_b216 = torch.as_tensor(np.asarray(load("z_in.bias"))).float()
cached_z216 = torch.cat([args216[4] for args216 in args_cache182])
print("POSTERIOR ANCHOR RMS",
      float(torch.mean(
          (cached_z216-original_z216[ids182])**2
      ).sqrt()))

def replay_action_token216(shifts, action_inputs, batch_size=128):
    global linear136
    original_linear = linear136
    shift_box = [None]
    def injected_linear(x, prefix):
        y = original_linear(x, prefix)
        return y + shift_box[0] if prefix == "z_in" else y
    result_h = torch.empty(n182, 4, 192)
    result_o = torch.empty(n182, 192)
    linear136 = injected_linear
    try:
        for positions in groups182.values():
            for offset in range(0, len(positions), batch_size):
                jj = positions[offset:offset+batch_size]
                packed = [
                    torch.cat([args_cache182[j][k] for j in jj], dim=0)
                    for k in range(7)
                ]
                packed[3] = action_inputs[ids182[jj]]
                shift_box[0] = shifts[jj]
                hh, oo = call_core161(tuple(packed), "key")
                result_h[jj], result_o[jj] = hh, oo
    finally:
        linear136 = original_linear
    return result_h, result_o

def candidate_readout216(vectors, scales=(1.0, 1.0)):
    u = encode_actions168(scales)
    z = private_posterior215(torch.cat([context, e_cal161, u], dim=1))
    token = F.linear(z, neutral_w216, neutral_b216)
    token = token + z[:, [16, 12]] @ torch.as_tensor(vectors.T).float()
    shifts = (token-original_token216)[ids182]
    hh, oo = replay_action_token216(shifts, u)
    means, logits = readout214(
        official_h182+hh-base_h214,
        official_o182+oo-base_o214
    )
    return array196(means), logits, array196(shifts)

candidates216 = {
    "row_ridge_0.1": np.column_stack([
        directions_fit215[("insulin", 0.1)],
        directions_fit215[("norepinephrine", 0.1)]
    ]),
    "mean_constraints": np.column_stack([
        mean_directions216["insulin"],
        mean_directions216["norepinephrine"]
    ])
}
results216 = {}
for label216, vectors216 in candidates216.items():
    factual_means216, factual_logits216, factual_shifts216 = (
        candidate_readout216(vectors216)
    )
    factual_change216 = factual_means216-array196(mean214)
    print("\nCANDIDATE", label216,
          "column norms", np.linalg.norm(vectors216, axis=0),
          "factual mean change", factual_change216.mean(0),
          "factual mean RMS", np.sqrt(np.mean(factual_change216**2, axis=0)),
          "token RMS", np.sqrt(np.mean(factual_shifts216**2)))
    results216[label216] = {"factual_means": factual_means216}
    for drug216, column216 in [("insulin", 0), ("norepinephrine", 1)]:
        select216 = drug_labels215 == drug216
        held216 = select216 & hold215
        for scale216 in [0.0, 0.5, 2.0, 3.0]:
            scales216 = [1.0, 1.0]
            scales216[column216] = scale216
            edited_means216, _, _ = candidate_readout216(
                vectors216, tuple(scales216)
            )
            change216 = edited_means216-factual_means216
            dr216 = (scale216-1.0)*rate_vector215
            slope216 = (
                (dr216[select216, None]*change216[select216]).sum(0)
                / np.sum(dr216[select216]**2)
            )
            print("FINITE CONTRAST", drug216, scale216,
                  "mean", change216[select216].mean(0),
                  "held mean", change216[held216].mean(0),
                  "per-rate slope", slope216)
            results216[label216][(drug216, scale216)] = change216

np.savez("/work/response_candidates216.npz", **candidates216)
print("SECONDS", round(time.time()-start216, 2))
with open("/work/notes.md", "a") as f216:
    f216.write(
        "\n## Python 216 — finite-dose private-token calibration checks\n"
        "- Local rowwise regression underdelivered because many original states ignore the latent token.\n"
        "- Compared rowwise and minimum-norm mean-response directions with actual x0, half, x2 and x3 action encodings.\n"
        "- Testing saturation, cross-sensor leakage and factual proxy shifts before installation.\n"
        "- Neutral209 remains the only installed edit.\n"
    )
```