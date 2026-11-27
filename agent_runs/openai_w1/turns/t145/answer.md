The records support negative insulin effects and positive norepinephrine effects, subject to the identification assumptions; the original norepinephrine rollout is much weaker. Older reconstruction notes also mention interleaved attention. I’ll check whether self- and cross-attention were decoded inconsistently. Near-exact state agreement would confirm that explanation; another large residual would reject it.

```python
import itertools, torch, numpy as np
from pathlib import Path

print("=== NEXT EARLY NOTEBOOK SECTION ===")
print(Path("/work/notes.md").read_text()[6200:10800])

print("\n=== FACTUAL HEAD CACHE ===")
for k, v in ce_head.items():
    print(k, type(v).__name__, getattr(v, "shape", None))
print("Cross helper uses layout switch:", "rx_layout" in rx_cross.__code__.co_names)

rx_cross_before133 = rx_cross
rx_layout_before133 = rx_layout
rx_cross_layout133 = "split"

def rx_cross133(x, m, age, valid, p, mode, age_scale, act):
    q = rx_linear(x, p + ".q")
    av = rx_mlp(torch.log1p(age.clamp_min(0)).unsqueeze(-1) / age_scale,
                "core.age", act)
    if mode == "before_kv":
        kv = rx_linear(m + av, p + ".kv")
        raw_kv = kv
    elif mode == "before_k":
        raw_kv = rx_linear(m, p + ".kv")
        kv = rx_linear(m + av, p + ".kv")
    else:
        raw_kv = kv = rx_linear(m, p + ".kv")
    B, L, D = q.shape
    S = m.shape[1]
    if rx_cross_layout133 == "split":
        k = kv[..., :D]
        v = raw_kv[..., D:]
    else:
        k = kv.reshape(B, S, 4, 2, 48)[..., 0, :].reshape(B, S, D)
        v = raw_kv.reshape(B, S, 4, 2, 48)[..., 1, :].reshape(B, S, D)
    if mode == "after_k":
        k = k + av
    a = torch.nn.functional.scaled_dot_product_attention(
        q.reshape(B, L, 4, 48).transpose(1, 2),
        k.reshape(B, S, 4, 48).transpose(1, 2),
        v.reshape(B, S, 4, 48).transpose(1, 2),
        attn_mask=valid[:, None, None, :]
    )
    return rx_linear(a.transpose(1, 2).reshape(B, L, D), p + ".out")

rx_cross = rx_cross133
rx_tests133 = []
for self_layout, cross_layout, mode, pre_act, block_act, coda_act in itertools.product(
    ("split", "interleaved"), ("split", "interleaved"),
    ("before_kv", "before_k", "after_k"),
    ("gelu", "silu"), ("gelu", "silu"), ("gelu", "silu")
):
    rx_layout = self_layout
    rx_cross_layout133 = cross_layout
    E = rx_norm(
        rx_layer2(rx_tokens126, "core.prelude.0", pre_act)[:, :4],
        "core.prelude_norm"
    )
    base = rx_hflow + E
    u = base.clone()
    for layer in ("core.blocks.0.0", "core.blocks.0.1"):
        u = rx_layer2(u, layer, block_act, rx_mem, rx_ages, rx_memvalid,
                      mode, "gelu", 1.0)
    h = rx_abar126[None, None, :] * rx_hflow + \
        rx_delta126[None, None, :] * rx_linear(E, "core.b_in") + u-base
    y = rx_layer2(rx_linear(h, "core.c_out"), "core.coda.0", coda_act)
    out = rx_norm(y.mean(1), "core.out_norm")
    rx_tests133.append((
        float((y-rx_targeth).square().mean().sqrt()),
        float((out-rx_targetout).square().mean().sqrt()),
        self_layout, cross_layout, mode, pre_act, block_act, coda_act
    ))
rx_tests133.sort()
print("\n=== JOINT SELF/CROSS LAYOUT TESTS ===")
for result in rx_tests133[:8]:
    print(result)

rx_cross = rx_cross_before133
rx_layout = rx_layout_before133
with open("/work/notes.md", "a") as f:
    f.write(
        "\n\n## Run 133\n"
        "Recovered early record evidence and tested independent self/cross "
        f"attention layouts. Best candidates: {rx_tests133[:3]}. "
        "No weights edited. Counts: 133 Python runs, 6 forward passes.\n"
    )
```