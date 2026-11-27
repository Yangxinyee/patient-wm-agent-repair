The whole-stay audit now matches: tuning value NLL is 1.63178 locally versus 1.63164 on the GPU, with negligible sensor-specific bias. That is sufficient to test a stochastic-safe correction. I’ll reconstruct paired dose contrasts using the same prior draws and fit only the final value-head layer, avoiding the unstable replacement units.

```python
print("=== ORIGINAL ROLLOUT MEANS ===")
d146 = dosesummary144[
    dosesummary144.reached & dosesummary144.hour.between(13, 24)]
print(d146.groupby(["edit", "stay_id"])["shift"].mean().groupby("edit").mean())

def wm_mc_raw146(meta, raw, noise):
    numbers = torch.zeros_like(raw["numbers"])
    numbers[:, 0] = torch.as_tensor(meta.has_last.to_numpy(), dtype=torch.float32)
    numbers[:, 3] = torch.log1p(torch.as_tensor(
        meta.age_s.to_numpy(), dtype=torch.float32)) / 10
    numbers[:, 6] = torch.as_tensor(meta.zlast.to_numpy(),
                                    dtype=torch.float32).clamp(-10, 10)
    inputs0 = raw["inputs"] + F.linear(
        numbers - raw["numbers"], wt("last_numeric.weight"), None)
    prior = mlp(raw["context"], "z_prior", "gelu")
    view = raw["view"][None] + F.linear(
        noise * prior[:, 32:].exp()[None], wt("z_head.weight"), None)
    q0 = ln(raw["view"] + wt("type_emb.weight")[1], "query_norm")
    inputs = inputs0[None] + ln(
        view + wt("type_emb.weight")[1], "query_norm") - q0[None]
    hidden = F.gelu(F.linear(
        inputs, wm_weights145["attr_heads.value_bin.0.weight"],
        wm_weights145["attr_heads.value_bin.0.bias"]))
    transition = wt("embed_transition.value_bin.weight")[
        raw["code"].long() * 17 + raw["last_bin"].long()]
    logits = F.linear(
        hidden, wm_weights145["attr_heads.value_bin.2.weight"],
        wm_weights145["attr_heads.value_bin.2.bias"]) + transition[None]
    probs = logits.softmax(-1)
    return dict(raw=raw, inputs0=inputs0, inputs=inputs, hidden=hidden,
                logits=logits, probs=probs,
                mean=(probs.mean(0) * wm_train145["centres"]).sum(-1))

wm_cf146 = {}
for key146, local146 in r117_cf.items():
    print("\n", key146, "fields:", list(local146))
    raw146 = local146.get("head", local146)
    print("head fields:", list(raw146))
    if not all(k in raw146 for k in
               ("view", "context", "inputs", "numbers", "code", "last_bin")):
        continue
    cf146 = wm_mc_raw146(hm, raw146, wm_train145["noise"])
    cf146["active"] = np.asarray(local146["active"], dtype=bool)
    cf146["delta_rate"] = np.asarray(local146["delta_rate"], dtype=float)
    inactive146 = ~cf146["active"]
    print("inactive input RMSE:",
          float((cf146["inputs0"][inactive146]
                 - wm_train145["inputs0"][inactive146]).square().mean().sqrt()))
    rows146 = []
    for ev146 in hm.event.unique():
        m146 = cf146["active"] & (hm.event.to_numpy() == ev146)
        if m146.any():
            delta146 = (cf146["mean"] - wm_train145["mean"]).numpy()[m146]
            rate146 = cf146["delta_rate"][m146]
            rows146.append((ev146, int(m146.sum()), float(delta146.mean()),
                            float(np.dot(rate146, delta146) /
                                  max(np.dot(rate146, rate146), 1e-12))))
    print(pd.DataFrame(rows146, columns=["event", "n", "mean_shift", "dose_slope"])
          .to_string(index=False))
    wm_cf146[key146] = cf146

print("\nRecord-derived immediate coefficients:", repair_drug_betas)
print("Built stochastic contrast groups:", list(wm_cf146))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 145–146: complete stochastic reconstruction\n"
            "- Rebuilt reading metadata agrees exactly with existing training metadata.\n"
            "- Four-draw-mixture CPU tuning NLL 1.631776 versus GPU 1.631642; scaled mean RMSE 0.0564, negligible bias.\n"
            "- Rebuilding common-noise local dose contrasts for a final-layer-only correction.\n"
            "- No edits; 14 forward passes used.\n")
```