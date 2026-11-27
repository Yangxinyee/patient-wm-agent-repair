The paired probes reach the value head, but the original responses are weak and often saturate at higher rates. I’ll build a larger, stay-separated calibration set for a value-head-only edit. This approach will be supported if it can recover dose sensitivity with little change to factual logits; a poor trade-off on held-out stays would refute it.

```python
from collections import defaultdict
import time

start182 = time.time()
rows_by_stay182 = defaultdict(list)
for group, row in zip(act_groups152, act_rows152):
    if 1 <= float(row["hour"]) < 24:
        rows_by_stay182[int(row["stay_id"])].append((int(group), dict(row)))

chosen182 = []
for stay, choices in sorted(rows_by_stay182.items()):
    positions = np.unique(np.linspace(0, len(choices)-1, min(4, len(choices))).astype(int))
    chosen182.extend(choices[j] for j in positions)
if len(chosen182) > 320:
    rng182 = np.random.default_rng(182)
    keep182 = np.sort(rng182.choice(len(chosen182), 320, replace=False))
    chosen182 = [chosen182[j] for j in keep182]

ids182 = np.array([x[0] for x in chosen182], dtype=int)
stays182 = stays156[ids182].astype(np.int64)
drugs182 = np.array([x[1]["event"] for x in chosen182])
rates182 = np.array([float(x[1]["rate"]) for x in chosen182])
n182 = len(ids182)
events182 = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
args_cache182 = [replay_inputs161(int(i), include_current=best_memory181[0],
                                  age_scale=best_memory181[1]) for i in ids182]
groups182 = defaultdict(list)
for j, args in enumerate(args_cache182):
    groups182[int(args[5].shape[1])].append(j)

oldpost182 = mlp136(torch.cat([context[ids182], e_cal161[ids182],
                              U179["base"][ids182]], 1), "z_post")[:, :32]

def batch_replay182(Uall):
    newpost = mlp136(torch.cat([context[ids182], e_cal161[ids182],
                               Uall[ids182]], 1), "z_post")[:, :32]
    result_h = torch.empty(n182, 4, 192)
    result_o = torch.empty(n182, 192)
    for length, js in groups182.items():
        for offset in range(0, len(js), 32):
            jj = js[offset:offset+32]
            packed = [torch.cat([args_cache182[j][k] for j in jj], 0)
                      for k in range(7)]
            packed[3] = Uall[ids182[jj]]
            packed[4] = packed[4] + newpost[jj] - oldpost182[jj]
            hh, oo = call_core161(tuple(packed), "key")
            result_h[jj], result_o[jj] = hh, oo
    return result_h, result_o

base_h182, base_o182 = batch_replay182(U179["base"])
official_h182 = state_cal161[ids182, 192:960].reshape(n182, 4, 192)
official_o182 = state_cal161[ids182, :192]

reg_terms182 = torch.empty(n182, 4, 192)
base_x182 = torch.empty(n182, 4, 192)
base_logits182 = torch.empty(n182, 4, 17)
gap_features182 = torch.empty(n182, 192)
gap_seconds182 = torch.empty(n182)
base_q182 = torch.empty(n182, 192)
for j, i in enumerate(ids182):
    for c, event in enumerate(events182):
        _, cap = capture159(official_h182[j:j+1], official_o182[j:j+1],
                            int(stays182[j]), float(hours156[i]), event)
        reg_terms182[j, c] = cap["with_code"][0] - cap["q"][0]
        base_x182[j, c] = F.gelu(linear136(cap["with_code"], "attr_heads.value_bin.0"))[0]
        base_logits182[j, c] = cap["logits"][0]
        if c == 0:
            gap_seconds182[j] = float(cap["gap_seconds"])
            gap_features182[j] = (cap["ctx"] - cap["out"]
                - norm136(cap["flowed"].mean(1), "core.out_norm"))[0]
            base_q182[j] = cap["q"][0]

def batch_q182(hh, oo):
    flowed = hh * torch.exp(-gap_seconds182[:, None, None] * torch.exp(W["core.flow.a"]))
    ctx = oo + norm136(flowed.mean(1), "core.out_norm") + gap_features182
    view = ctx + linear136(mlp136(ctx, "z_prior")[:, :32], "z_head")
    return norm136(view + W["type_emb.weight"][1], "query_norm")

print("calibration bundles/stays:", n182, len(np.unique(stays182)))
print("base q reconstruction max error:",
      float((batch_q182(official_h182, official_o182)-base_q182).abs().max()))

centres182 = torch.stack([
    torch.cat([torch.zeros(1), torch.as_tensor(centres16_156(e)).reshape(-1)])
    for e in events182
])
tilt182 = torch.cat([torch.zeros(1), torch.linspace(-1., 1., 16)])
base_p182 = torch.softmax(base_logits182, -1)
base_mean182 = (base_p182 * centres182[None]).sum(-1)
sensitivity182 = (
    base_p182 * (centres182[None]-base_mean182[:, :, None]) * tilt182
).sum(-1).clamp_min(.05)

contrasts182 = []
for scale in [0., .5, 2., 3.]:
    for drug in ["insulin", "norepinephrine"]:
        scales = (scale, 1.) if drug == "insulin" else (1., scale)
        Unew = torch.as_tensor(encode_actions168(scales), dtype=torch.float32)
        hh, oo = batch_replay182(Unew)
        qnew = batch_q182(official_h182+hh-base_h182,
                          official_o182+oo-base_o182)
        xnew = F.gelu(linear136(qnew[:, None, :] + reg_terms182,
                               "attr_heads.value_bin.0"))
        new_logits = base_logits182 + F.linear(
            xnew-base_x182, W["attr_heads.value_bin.2.weight"])
        new_mean = (torch.softmax(new_logits, -1) * centres182[None]).sum(-1)
        true_shift = torch.zeros(n182, 4)
        selected = drugs182 == drug
        delta_rate = torch.as_tensor((scale-1.) * rates182, dtype=torch.float32)
        if drug == "insulin":
            true_shift[selected, :2] = -1.99 * delta_rate[selected, None]
        else:
            true_shift[selected, 2:] = 3.29 * delta_rate[selected, None]
        target_tilt = (true_shift - (new_mean-base_mean182)) / sensitivity182
        contrasts182.append({
            "drug": drug, "scale": scale,
            "dx": (xnew-base_x182).numpy(),
            "target": target_tilt.numpy(),
            "true_shift": true_shift.numpy(),
            "original_shift": (new_mean-base_mean182).numpy()
        })
        own = slice(0, 2) if drug == "insulin" else slice(2, 4)
        print(drug, scale,
              "mean target/original", round(float(true_shift[selected, own].mean()), 3),
              round(float((new_mean-base_mean182)[selected, own].mean()), 3),
              "contrast RMS", round(float((xnew-base_x182)[selected, own].square().mean().sqrt()), 6))

print("elapsed seconds:", round(time.time()-start182, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 182: direct value-head calibration data\n"
            "- Built paired anchored head features for doses x0, x0.5, x2, x3 on training states.\n"
            "- Targets are record-derived immediate slopes (-1.99 insulin, +3.29 NE), with\n"
            "  zero cross-drug sensor changes. These are calibration assumptions, not truth labels.\n"
            "- Candidate will be fitted by regression and judged by held-stay factual preservation\n"
            "  and actual GPU rollouts before acceptance.\n")
```
