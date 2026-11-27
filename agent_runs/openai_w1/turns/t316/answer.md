The audit found that the local replay helpers were using attention weights different from the installed edit. My reconstruction also mishandled stay-boundary time gaps, causing overflow. I’ll rebuild the features with explicit weights and rounded-minute gaps. Matching the verified time-flow states and exported posterior means would support using the new cache.

```python
import numpy as np, torch
import torch.nn.functional as F

W277 = {n: load(n).detach().cpu().float() for n in NAMES}
O277 = {n: load_original(n).detach().cpu().float() for n in NAMES}

def lin277(x, p, w=W277):
    return F.linear(x, w[p+".weight"], w.get(p+".bias"))

def norm277(x, p, w=W277):
    return F.layer_norm(x, (192,), w[p+".weight"], w[p+".bias"])

def mlp277(x, p, w=W277):
    return lin277(F.gelu(lin277(x, p+".0", w)), p+".2", w)

def prelude277(raw, w=W277):
    x = raw
    n = norm277(x, "core.prelude.0.norm1", w)
    b, length, _ = n.shape
    q = lin277(n, "core.prelude.0.attn.q", w).reshape(b, length, 4, 48).transpose(1, 2)
    k, v = lin277(n, "core.prelude.0.attn.kv", w).chunk(2, -1)
    k = k.reshape(b, length, 4, 48).transpose(1, 2)
    v = v.reshape(b, length, 4, 48).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, length, 192)
    x = x + lin277(a, "core.prelude.0.attn.out", w)
    x = x + mlp277(norm277(x, "core.prelude.0.norm3", w), "core.prelude.0.ffn", w)
    return norm277(x[:, :4], "core.prelude_norm", w)

minute277 = np.array([k[1] for k in rr_keys4], dtype=np.float32)
same277 = np.array([rr_keys4[i][0] == rr_keys4[int(prev276[i])][0]
                    and int(prev276[i]) != i for i in range(len(rr_keys4))])
dt277 = np.where(same277, np.maximum(0, minute277-minute277[prev276])*60, 0).astype(np.float32)
hflow277 = torch.as_tensor(rr_prevh4, dtype=torch.float32) * torch.exp(
    -torch.as_tensor(dt277)[:, None, None] * torch.exp(O277["core.flow.a"])[None, None])
slots277 = hflow277 + O277["core.slots"]
e277 = torch.as_tensor(rr_test_e111, dtype=torch.float32)
u277 = torch.as_tensor(rr_test_u111, dtype=torch.float32)
ctx277 = torch.as_tensor(rr_best_ctx111, dtype=torch.float32)
inp277 = torch.cat([ctx277, e277, u277], 1)
mu277 = mlp277(inp277, "z_post")[:, :32]
muorig277 = mlp277(inp277, "z_post", O277)[:, :32]
zin277 = lin277(mu277, "z_in")
raw277 = torch.cat([slots277, e277[:, None], u277[:, None], zin277[:, None]], 1)
Ef277 = torch.cat([prelude277(x) for x in raw277.split(512)]).numpy()

print("=== CORRECTED RECONSTRUCTION CHECKS ===")
print("all inputs finite:", bool(torch.isfinite(raw277).all()), bool(np.isfinite(Ef277).all()))
print("rounded dt match:", float(np.max(np.abs(dt277[pick276]-np276(rx_dt)))))
print("hflow versus verified RMS:",
      float(np.sqrt(np.mean((hflow277[pick276].numpy()-np276(rx_hflow))**2))))
print("cached slot discrepancy, valid bundles:",
      float(np.sqrt(np.mean((np276(slots258)[same277]-slots277.numpy()[same277])**2))))
print("original posterior versus exported RMS, valid:",
      float(np.sqrt(np.mean((muorig277.numpy()[same277]-rr_zexport4[same277])**2))))
check277 = torch.as_tensor(raw272, dtype=torch.float32)
oldcheck277 = prelude269(check277)
newcheck277 = prelude277(check277, {n: torch.as_tensor(np276(rx_W[n])) for n in NAMES})
print("independent prelude versus existing exact helper RMS:",
      float(torch.sqrt(torch.mean((oldcheck277-newcheck277)**2))))
print("corrected E RMS:", float(np.sqrt(np.mean(Ef277**2))))
print("old P266 factual projection RMS:",
      np.sqrt(np.mean((Ef277.reshape(-1, 192) @ fits266[1.0])**2, axis=0)))

print("\n=== COUNTERFACTUAL CACHE DICTIONARIES ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if isinstance(o, dict) and ("insulin", 2.0) in o:
        val = o[("insulin", 2.0)]
        print(n, list(o.keys())[:4], "example shape", getattr(val, "shape", None))

print("\n=== PHYSICAL DOSE OBJECTS ===")
for n in sorted(list(globals())):
    if n.startswith("ds_"):
        o = globals()[n]
        if isinstance(o, dict):
            print(n, "dict keys", list(o.keys())[:8])
        elif hasattr(o, "shape"):
            print(n, tuple(o.shape))
        elif callable(o) and hasattr(o, "__code__"):
            print(n, "fn", o.__code__.co_varnames[:o.__code__.co_argcount])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 277\nRebuilt prelude independently using explicit installed weights, rounded-minute time gaps and stay-boundary masking. Cached residual slots and stale replay weights were potential sources of the failed injections. Checking finite features, verified flow and exported posterior alignment before refitting. No weights changed. Counts: 277 Python runs, 29 forwards.\n")
```