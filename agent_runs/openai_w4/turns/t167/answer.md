The action reconstruction matches the retained factual encoding, but the insulin summary was spoiled by non-finite entries. It therefore gives no usable compression result yet. I’ll restrict the comparison to contexts that are finite under every multiplier. The norepinephrine changes measured here are leakage into the insulin channel, not evidence of a norepinephrine repair.

```python
import numpy as np
import torch
import dis
import io

print("=== FINITE-CONTEXT CHANNEL RESPONSES ===")
for drug148, obj148 in channel_probe147.items():
    res148 = obj148["results"]
    rate148 = obj148["rate"]
    good148 = np.isfinite(rate148)
    for v148 in res148.values():
        good148 &= np.isfinite(v148["q"])
    print("\n", drug148, "finite:", int(good148.sum()), "/", len(good148))
    if not good148.any():
        continue
    qbase148 = res148[1.0]["q"][good148]
    r148 = rate148[good148]
    print("rate quantiles:", np.quantile(r148, [0, .1, .5, .9, 1]).round(4))
    print("factual q quantiles:", np.quantile(qbase148, [0, .1, .5, .9, 1]).round(4))
    for factor148 in [0.0, 0.5, 2.0, 3.0, 4.0]:
        dr148 = (factor148 - 1.0) * r148
        dq148 = res148[factor148]["q"][good148] - qbase148
        nz148 = dr148 != 0
        slope148 = float(dr148 @ dq148 / max(dr148 @ dr148, 1e-12))
        print("factor", factor148, "mean dq", round(float(dq148.mean()), 5),
              "slope", round(slope148, 5),
              "dq/dr quantiles",
              np.quantile(dq148[nz148] / dr148[nz148], [.1, .5, .9]).round(4))
        if drug148 == "insulin" and factor148 in [0.5, 2.0]:
            for lo148, hi148 in [(0, 1.5), (1.5, 3), (3, 6), (6, np.inf)]:
                g148 = (r148 > lo148) & (r148 <= hi148)
                if g148.any():
                    print("  rate band", (lo148, hi148), "n", int(g148.sum()),
                          "mean dose change", round(float(dr148[g148].mean()), 4),
                          "mean q change", round(float(dq148[g148].mean()), 4))

print("\n=== CHANNEL PARAMETER SHAPES ===")
for name148 in ["z_post.0.weight", "z_post.2.weight", "z_in.weight",
                "core.prelude.0.attn.q.weight", "core.b_in.weight"]:
    print(name148, SHAPES[name148])
print("gated_innovation94 locals:", gated_innovation94.__code__.co_varnames)
print("gated_innovation94 dependencies:", gated_innovation94.__code__.co_names)
stream148 = io.StringIO()
dis.dis(gated_innovation94, file=stream148)
print(stream148.getvalue()[:2400])

print("\n=== CHANNEL-CONSTRUCTION FUNCTIONS ===")
for k148, v148 in sorted(list(globals().items())):
    if callable(v148) and hasattr(v148, "__code__"):
        if any(s148 in k148.lower() for s148 in
               ["candidate", "construct", "wire", "innovation", "response", "direction"]):
            print(k148, v148.__code__.co_varnames[:8])

with open("/work/notes.md", "a") as f148:
    f148.write(
        "\n## Run 148\n"
        "Repeated channel summaries on contexts finite under all multipliers. "
        "Run-147 insulin NaN summaries were invalid; no compression conclusion was drawn. "
        "No weights changed.\n"
    )
```