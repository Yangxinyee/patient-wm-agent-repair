The replay instructions show where to test a correction: after the prelude, at `core.b_in`, rather than through a larger latent token. I’ll build a replay that accepts a tensor directly and exposes that intermediate state. It is usable only if its original-weight outputs match official state dumps.

```python
import numpy as np, torch
import torch.nn.functional as F

print("=== REPLAY DEPENDENCIES ===")
for n228 in ("lin0", "mlp0", "norm0", "audit_block"):
    f228 = globals()[n228]
    print(n228, "args", f228.__code__.co_varnames[:f228.__code__.co_argcount],
          "defaults", brief226(f228.__defaults__),
          "names", f228.__code__.co_names)

def linear228(x, name):
    return F.linear(x, W0[name + ".weight"], W0.get(name + ".bias"))

# Restore an unambiguous original-weight linear helper for diagnostics only.
old_lin228 = lin0
lin0 = linear228

def core228(token, u=None, correction=None, include_current=True):
    c = context151
    h, e = c["h"], c["e"]
    u = c["u"] if u is None else u
    token = torch.as_tensor(token, dtype=torch.float32)
    assert token.shape == e.shape, (token.shape, e.shape)
    memory, mask, ages = c["memory"], c["mask"], c["ages"]
    if include_current:
        memory = torch.cat([memory, e[:, None]], 1)
        mask = torch.cat([mask, torch.ones((len(h), 1), dtype=torch.bool)], 1)
        ages = torch.cat([ages, torch.zeros((len(h), 1, 1))], 1)
    tokens = torch.cat([h + W0["core.slots"][None],
                        e[:, None], u[:, None], token[:, None]], 1)
    E = norm0(audit_block(tokens, "core.prelude.0")[:, :4],
              "core.prelude_norm")
    mem = memory + mlp0(torch.log1p(ages), "core.age", "gelu")
    state = audit_block(h + E, "core.blocks.0.0", mem, mask)
    state = audit_block(state, "core.blocks.0.1", mem, mask)
    delta = F.softplus(W0["core.delta_in"])
    retention = torch.exp(-delta * torch.exp(W0["core.a_in"]))
    pre = retention[None, None] * h + delta[None, None] * linear228(E, "core.b_in")
    pre = pre + state - (h + E)
    if correction is not None:
        pre = pre + delta[None, None] * F.linear(E, correction)
    hnew = audit_block(linear228(pre, "core.c_out"), "core.coda.0")
    out = norm0(hnew.mean(1), "core.out_norm")
    return {"out": out, "h": hnew, "E": E, "pre": pre}

post228 = mlp0(torch.cat([context151["ctx"], context151["e"],
                         context151["u"]], -1), "z_post")
token228 = linear228(post228[:, :32], "z_in")
base228 = core228(token228)
print("original local shapes:", {k: tuple(v.shape) for k, v in base228.items()})
print("original local RMS:", {k: round(float(v.square().mean().sqrt()), 5)
                              for k, v in base228.items()})

print("=== CONTEXT INDEX / OFFICIAL STATE OBJECTS ===")
for n228 in sorted(list(globals())):
    if ("151" in n228 or n228.startswith(("r49_", "rr33_"))) and not callable(globals()[n228]):
        o228 = globals()[n228]
        if hasattr(o228, "shape"):
            print(n228, tuple(o228.shape))
        elif isinstance(o228, (list, tuple, dict)):
            print(n228, type(o228).__name__, len(o228),
                  list(o228.keys())[:5] if isinstance(o228, dict) else "")
        elif isinstance(o228, (str, int, float)):
            print(n228, repr(o228)[:100])

print("=== RECORD ESTIMATES ===")
for p228 in open("/work/notes.md").read().split("\n\n"):
    if p228.startswith(("## Run 5:", "## Run 6:", "## Run 7:")):
        print(p228[:1600])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 228: explicit prelude/recurrent replay\n"
            "- Built tensor-only replay exposing E and the pre-coda recurrent state.\n"
            "- Restored an original-weight local linear helper; submitted weights unchanged.\n"
            "- Replay must match official original states before it guides core.b_in edits.\n"
            "- Python runs: 228; forward passes: 17.\n")
```