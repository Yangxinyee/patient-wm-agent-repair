The replay has noticeable absolute state error, so I won’t treat its predictions as certified. The readout helper is usable, however. I’ll test paired dose changes anchored to official states and compare memory conventions. A direct head correction is worth pursuing only if those contrasts are consistent across stays and reach the intended sensor outputs.

```python
drug_probes181 = {}
for drug in ["insulin", "norepinephrine"]:
    choices = [(int(g), dict(r)) for g, r in zip(act_groups152, act_rows152)
               if r["event"] == drug and 4 <= float(r["hour"]) <= 22]
    positions = np.linspace(0, len(choices)-1, 4).astype(int)
    drug_probes181[drug] = [choices[j] for j in positions]

print("=== MEMORY CONVENTION CHECK ===")
errors181 = {}
for include in [False, True]:
    for age_scale in [1., 1./3600.]:
        vals = []
        for drug in drug_probes181:
            for i, row in drug_probes181[drug][:2]:
                args = replay_inputs161(i, include_current=include, age_scale=age_scale)
                hh, oo = call_core161(args, "key")
                target_h = state_cal161[i:i+1, 192:960].reshape(1, 4, 192)
                target_o = state_cal161[i:i+1, :192]
                vals.append((
                    float((hh-target_h).square().mean().sqrt()
                          / target_h.square().mean().sqrt()),
                    float((oo-target_o).square().mean().sqrt())
                ))
        errors181[(include, age_scale)] = np.mean(vals, axis=0)
        print(include, age_scale, errors181[(include, age_scale)])
best_memory181 = min(errors181, key=lambda k: errors181[k][1])
print("selected diagnostic convention:", best_memory181)

print("\n=== ANCHORED DOSE CONTRASTS ===")
probe_pairs181 = []
for drug, choices in drug_probes181.items():
    label = "insulin2" if drug == "insulin" else "ne2"
    for i, row in choices:
        args = list(replay_inputs161(i, include_current=best_memory181[0],
                                    age_scale=best_memory181[1]))
        h0, o0 = call_core161(tuple(args), "key")
        old_post = mlp136(torch.cat([
            context[i:i+1], e_cal161[i:i+1], U179["base"][i:i+1]], 1), "z_post")[:, :32]
        new_post = mlp136(torch.cat([
            context[i:i+1], e_cal161[i:i+1], U179[label][i:i+1]], 1), "z_post")[:, :32]
        post_error = float((old_post-state_cal161[i:i+1, 960:992]).square().mean().sqrt())
        args[3] = U179[label][i:i+1]
        args[4] = args[4] + new_post - old_post
        h1, o1 = call_core161(tuple(args), "key")
        official_h = state_cal161[i:i+1, 192:960].reshape(1, 4, 192)
        official_o = state_cal161[i:i+1, :192]
        pairs = {}
        for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
            m0, c0 = capture159(official_h, official_o, int(stays156[i]),
                                float(hours156[i]), event)
            m1, c1 = capture159(official_h+h1-h0, official_o+o1-o0,
                                int(stays156[i]), float(hours156[i]), event)
            x0 = F.gelu(linear136(c0["with_code"], "attr_heads.value_bin.0"))
            x1 = F.gelu(linear136(c1["with_code"], "attr_heads.value_bin.0"))
            pairs[event] = (x0, x1, c0, c1)
            if event == ("glucose_finger" if drug == "insulin" else "map_arterial"):
                print(drug, "stay", int(stays156[i]), "hour", float(hours156[i]),
                      "rate", row["rate"], "posterior_error", round(post_error, 5),
                      "hidden_contrast_RMS", round(float((x1-x0).square().mean().sqrt()), 6),
                      "original_mean_shift", round(float(m1-m0), 4))
        probe_pairs181.append((drug, i, row, pairs))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 181: anchored dose-contrast diagnostic\n"
            "- Compared memory conventions and measured action-induced value-head feature changes.\n"
            "- Absolute replay error is not accepted as factual validation; only paired differences\n"
            "  anchored to official states will be used for a candidate head edit.\n")
```
