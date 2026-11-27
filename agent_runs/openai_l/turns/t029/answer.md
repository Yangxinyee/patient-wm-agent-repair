The official metric averages reached stay-hours. On that weighting, the provisional record targets are about −12.8/+6.4 for insulin and +4.5/−2.3 for norepinephrine. Query type and status did not resolve the head mismatch. I’ll test latent-sampling noise and the register layout; matching the GPU within sampling noise would validate the reconstruction.

```python
import itertools

# Quantify how much mismatch four prior samples can explain.
mc_checks25 = []
unique_specs25 = []
for candidate25 in head_tests24:
    key25 = candidate25[2:]
    if key25[-1] == 0:
        continue
    if key25 not in unique_specs25:
        unique_specs25.append(key25)
    if len(unique_specs25) == 5:
        break

for pa25, au25, ha25, ti25, st25 in unique_specs25:
    ctx25 = out22 + flow_summary22 + mlp22(
        torch.tensor(np.log(dt22)/10., dtype=torch.float32)[:, None], "gap", pa25)
    raw25 = mlp22(ctx25, "z_prior", pa25)
    age25 = np.log1p(age_hours22 * (3600 if au25 == "seconds" else 1))/10.
    nf25 = np.zeros((len(ds22), 8), np.float32)
    nf25[:, 0], nf25[:, 1], nf25[:, 2], nf25[:, 3] = has22, age25, last_z22, has22
    constant25 = (base_no_status24 + lin22(torch.tensor(nf25), "last_numeric")
                  + (0. if st25 == -1 else w22("last_status.weight")[st25]))
    gen25 = torch.Generator().manual_seed(2531)
    psum25 = torch.zeros(len(ds22), 17)
    mean_sum25, mean_sq25 = torch.zeros(len(ds22)), torch.zeros(len(ds22))
    for chunk25 in range(8):
        eps25 = torch.randn(8, len(ds22), 32, generator=gen25)
        zz25 = raw25[:, :32][None] + eps25 * raw25[:, 32:].exp()[None]
        qq25 = norm22(ctx25[None] + lin22(zz25, "z_head")
                      + w22("type_emb.weight")[ti25], "query_norm")
        logits_mc25 = mlp22(qq25 + constant25[None], "attr_heads.value_bin", ha25)
        pp25 = (logits_mc25 + transition22[None]).softmax(-1)
        mm25 = (pp25[:, :, 1:] * centres22[None]).sum(-1)
        psum25 += pp25.sum(0)
        mean_sum25 += mm25.sum(0)
        mean_sq25 += (mm25**2).sum(0)
    pbar25 = psum25 / 64
    mbar25 = mean_sum25 / 64
    var25 = (mean_sq25 / 64 - mbar25**2).clamp_min(0)
    nllbar25 = -pbar25[torch.arange(len(ds22)), observed_bin22].log()
    rmse25 = np.sqrt(np.mean((mbar25.numpy() - reference_mean22)**2))
    noise25 = np.sqrt(var25.mean().item() / 4)
    nllmae25 = np.mean(np.abs(nllbar25.numpy() - reference_nll22))
    result25 = (rmse25, noise25, nllmae25, pa25, au25, ha25, ti25, st25)
    mc_checks25.append(result25)
print("=== MC CHECKS: RMSE, EXPECTED FOUR-SAMPLE NOISE, NLL MAE, SPEC ===")
for r25 in sorted(mc_checks25):
    print(r25)

# Search plausible register permutations efficiently in head preactivation space.
# This only reconstructs original inputs; it changes no model weights.
screen25 = np.arange(450)
ctx25 = out22 + flow_summary22 + mlp22(
    torch.tensor(np.log(dt22)/10., dtype=torch.float32)[:, None], "gap", "silu")
raw25 = mlp22(ctx25, "z_prior", "silu")
q25 = norm22(ctx25 + lin22(raw25[:, :32], "z_head")
             + w22("type_emb.weight")[1], "query_norm")
fixed_first25 = lin22(q25 + base_no_status24 + w22("last_numeric.bias"),
                      "attr_heads.value_bin.0")[screen25]
projected_register25 = w22("attr_heads.value_bin.0.weight") @ w22("last_numeric.weight")
zz_screen25 = torch.tensor(last_z22[screen25])[:, None]
has_screen25 = torch.tensor(has22[screen25])[:, None]
centre_screen25 = centres22[screen25]
transition_screen25 = transition22[screen25]
reference_screen25 = reference_mean22[screen25]
register_tests25 = []

for age_unit25 in ["seconds", "hours"]:
    age_screen25 = torch.tensor(
        np.log1p(age_hours22[screen25] * (3600 if age_unit25 == "seconds" else 1))/10.,
        dtype=torch.float32)[:, None]
    for zpos25 in range(8):
        for agepos25 in range(8):
            if agepos25 == zpos25:
                continue
            available25 = [i for i in range(8) if i not in [zpos25, agepos25]]
            flag_options25 = [(i,) for i in available25]
            flag_options25 += list(itertools.combinations(available25, 2))
            for flags25 in flag_options25:
                ff25 = fixed_first25 + zz_screen25 * projected_register25[:, zpos25]
                ff25 = ff25 + age_screen25 * projected_register25[:, agepos25]
                ff25 = ff25 + has_screen25 * projected_register25[:, list(flags25)].sum(1)
                pp25 = (lin22(F.silu(ff25), "attr_heads.value_bin.2")
                        + transition_screen25).softmax(-1)
                mm25 = (pp25[:, 1:] * centre_screen25).sum(-1).numpy()
                err25 = np.sqrt(np.mean((mm25 - reference_screen25)**2))
                register_tests25.append((err25, age_unit25, zpos25, agepos25, flags25))
register_tests25.sort()
print("\n=== BEST REGISTER PERMUTATIONS ===")
for r25 in register_tests25[:15]:
    print(r25)

record_row_targets25 = {}
for e25, drug25, factor25 in [
    ("insulin_x2", "insulin", 1.), ("insulin_half", "insulin", -.5),
    ("ne_x2", "norepinephrine", 1.), ("ne_half", "norepinephrine", -.5)
]:
    r25 = roll22.loc[roll22.edit.eq(e25) & roll22.reached]
    vals25 = [factor25 * target_paths22[drug25][tune_pos22[s], int(h)]
              for s, h in zip(r25.stay_id, r25.hour)]
    record_row_targets25[e25] = float(np.mean(vals25))
print("\nOfficial-weighting provisional targets:", record_row_targets25)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 25\n"
            "Official readout is reached-row mean, not mean of stay means. "
            "Provisional record targets: " + repr(record_row_targets25) + ". "
            "Tested latent MC noise versus GPU mismatch: " + repr(sorted(mc_checks25)[0]) + ". "
            "Best register permutation: " + repr(register_tests25[0]) + ". "
            "No edits.\n")
```
