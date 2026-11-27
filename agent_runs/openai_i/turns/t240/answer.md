The estimated effects have substantial multi-hour tails, especially for insulin. The private coordinates have plausible innovation-scale variation, but their units still need checking. Next I’ll verify the anchored replay and test a small token perturbation. Agreement with cached readouts and a stable finite-difference response would justify fitting response directions.

```python
import time
import inspect

torch.set_num_threads(8)
start214 = time.time()

# Local calculations use original weights; installed edits are unchanged.
Wcal214 = {
    name214: torch.as_tensor(np.asarray(load_original(name214))).float()
    for name214 in NAMES
}
W = Wcal214

print("EVENT ORDER", events182)
print("ACTION ENCODER", inspect.signature(encode_actions168),
      encode_actions168.__code__.co_varnames,
      encode_actions168.__code__.co_names,
      str(encode_actions168.__code__.co_consts)[:500])

print("PRIVATE-COORDINATE RECONSTRUCTION")
if np.shape(feature_pred201) == (len(private_z209), 2):
    print("RMS versus fitted gated innovations",
          np.sqrt(np.mean(
              (array196(private_z209)[:, [16, 12]]
               - array196(feature_pred201))**2, axis=0)))

trans214 = (
    base_logits182
    - linear136(base_x182, "attr_heads.value_bin.2")
).clone()
centres214 = torch.as_tensor(array196(centres182)).float()

def replay_token214(shifts, indices=None, batch_size=128):
    """Original cached bundle replay with an additive latent-token perturbation."""
    global linear136
    if indices is None:
        indices = np.arange(n182)
    indices = np.asarray(indices, dtype=int)
    shifts = torch.as_tensor(shifts).float()
    assert shifts.shape == (len(indices), 192)
    result_h = torch.empty(len(indices), 4, 192)
    result_o = torch.empty(len(indices), 192)
    groups = {}
    for position, index in enumerate(indices):
        length = args_cache182[int(index)][5].shape[1]
        groups.setdefault(length, []).append(position)
    original_linear = linear136
    shift_box = [None]
    def injected_linear(x, prefix):
        y = original_linear(x, prefix)
        return y + shift_box[0] if prefix == "z_in" else y
    linear136 = injected_linear
    try:
        for positions in groups.values():
            for offset in range(0, len(positions), batch_size):
                pp = positions[offset:offset + batch_size]
                packed = tuple(
                    torch.cat([
                        args_cache182[int(indices[p])][k] for p in pp
                    ], dim=0)
                    for k in range(7)
                )
                shift_box[0] = shifts[pp]
                hh, oo = call_core161(packed, "key")
                result_h[pp] = hh
                result_o[pp] = oo
    finally:
        linear136 = original_linear
    return result_h, result_o

def readout214(hh, oo, indices=None):
    if indices is None:
        indices = np.arange(n182)
    indices = np.asarray(indices, dtype=int)
    flowed = hh * torch.exp(
        -gap_seconds182[indices, None, None]
        * torch.exp(W["core.flow.a"])
    )
    ctx = (
        oo + norm136(flowed.mean(1), "core.out_norm")
        + gap_features182[indices]
    )
    view = ctx + linear136(
        mlp136(ctx, "z_prior")[:, :32], "z_head"
    )
    q = norm136(
        view + W["type_emb.weight"][1], "query_norm"
    )
    logits = (
        mlp136(q[:, None, :] + reg_terms182[indices],
               "attr_heads.value_bin")
        + trans214[indices]
    )
    means = (logits.softmax(-1) * centres214[None, :, :]).sum(-1)
    return means, logits

base_h214, base_o214 = replay_token214(torch.zeros(n182, 192))
print("CACHE REPLAY RMS",
      float(torch.mean((base_h214-base_h182)**2).sqrt()),
      float(torch.mean((base_o214-base_o182)**2).sqrt()))
mean214, logits214 = readout214(official_h182, official_o182)
print("BASE READOUT CHECK",
      "mean RMS", float(torch.mean((mean214-base_mean182)**2).sqrt()),
      "logit RMS", float(torch.mean((logits214-base_logits182)**2).sqrt()))

rng214 = np.random.default_rng(214)
direction214 = rng214.normal(size=192)
direction214 -= direction214.mean()
direction214 /= np.linalg.norm(direction214)
perturb_results214 = {}
for epsilon214 in [0.02, -0.02, 0.04]:
    hh214, oo214 = replay_token214(
        torch.as_tensor(
            np.tile(epsilon214 * direction214, (n182, 1))
        ).float()
    )
    mm214, _ = readout214(
        official_h182 + hh214 - base_h214,
        official_o182 + oo214 - base_o214
    )
    perturb_results214[epsilon214] = array196(mm214 - mean214)
print("RANDOM-DIRECTION DERIVATIVES",
      np.mean(perturb_results214[0.02] / 0.02, axis=0),
      np.mean(perturb_results214[-0.02] / -0.02, axis=0),
      np.mean(perturb_results214[0.04] / 0.04, axis=0))
print("finite-difference relative disagreement",
      np.linalg.norm(
          perturb_results214[0.02] + perturb_results214[-0.02]
      ) / max(np.linalg.norm(perturb_results214[0.02]), 1e-12))

print("NATIVE DOSE-CONTRAST SUMMARIES")
for contrast214 in contrasts182:
    print(contrast214["drug"], contrast214["scale"],
          "original_shift shape", np.shape(contrast214["original_shift"]),
          "mean", np.mean(array196(contrast214["original_shift"]), axis=0))

print("SECONDS", round(time.time()-start214, 2))
with open("/work/notes.md", "a") as f214:
    f214.write(
        "\n## Python 214 — anchored token-response helper\n"
        "- Built forward-only latent-token perturbation and anchored readout helpers using original cached weights.\n"
        "- Installed six neutral209 tensors unchanged; no response columns added.\n"
        "- Checking baseline agreement and finite-difference stability before closed-form direction fitting.\n"
    )
```