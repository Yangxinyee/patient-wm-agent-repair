The recovered features explain part of the asymmetry: halving norepinephrine produces only about one eighth of its doubling contrast, and the insulin feature nearly saturates between x2 and x3. The MAP route also has roughly ten times the glucose route’s weight norm. I’ll test the glucose route alone at full strength. Passing factual loss with a stronger insulin response would support retaining it; MAP degradation would show that the routes are not sufficiently isolated.

```python
r256_strength_I = 1.0
r256_strength_N = 0.0
r256_C = r256_strength_I*r255_CI + r256_strength_N*r255_CN

def r256_predict(case, correction, strength=1.0):
    ee = np242(case["E"])
    inj = np.einsum("bsi,ji->bsj", ee, correction) * r245_delta[None, None, :]
    pp = case["pre"] + torch.as_tensor(strength*inj, dtype=torch.float32)
    return np242(r244_read_pre(pp, indices151))

r256_fact0 = np242(r244_read_pre(r245_cases[("factual", 1.0)]["pre"], indices151))
r256_fact1 = r256_predict(r245_cases[("factual", 1.0)], r256_C)
print("LOCAL factual mean changes, finger/lab/arterial/cuff:",
      np.mean(r256_fact1-r256_fact0, axis=0).round(4).tolist())
print("LOCAL factual RMS changes:",
      np.sqrt(np.mean((r256_fact1-r256_fact0)**2, axis=0)).round(4).tolist())
for drug in ["insulin", "norepinephrine"]:
    for mult in [0.0, 0.5, 1.5, 2.0, 3.0]:
        case = r245_cases[(drug, mult)]
        pr = r256_predict(case, r256_C)
        print("LOCAL", drug, mult, "vs factual:",
              np.mean(pr-r256_fact1, axis=0).round(4).tolist())

r256_weight = np242(r254_original) + r256_C
save_edit("core.b_in.weight", torch.as_tensor(r256_weight, dtype=torch.float32))
print("Submitted glucose-only route; edited tensors:", edited())
print("STAYS FOR FORWARD:", json.dumps([int(x) for x in r251_stays]))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 256: full-strength glucose-only recurrent candidate\n"
        "- Split correction into glucose route (norm 118.13) and MAP route (norm 1169.18).\n"
        "- Current feature contrasts are nonlinear: insulin x2/x3 ~0.304/0.323; "
        "NE x2/half ~0.03174/-0.00397, explaining weak half-dose response.\n"
        "- Submitted core.b_in = original + glucose route, with MAP route disabled.\n"
        "- Local multiplier checks are diagnostic only; official factual and rollout validation next.\n"
        "- Python runs: 256; forward passes: 19.\n"
    )
```
