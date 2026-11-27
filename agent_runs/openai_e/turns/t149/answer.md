The helpers use original weights, and the edit composition is now verified. The reconstruction error is about 10% of the hidden-state scale, so I suspect a forward-calculation detail, particularly memory-age handling. I’ll test a small set of interpretations against stored GPU states. A near-zero error would identify the correct reconstruction; otherwise I’ll avoid treating it as exact.

```python
import numpy as np, torch
from torch.nn import functional as F

print("attention defaults:", core_attn.__defaults__)
print("layer defaults:", core_layer.__defaults__)
print("age tensors:", [(k, tuple(v.shape)) for k,v in W0.items()
                        if k.startswith("core.age")])
print("FFN constants:", core_ffn.__code__.co_consts)

oldrows129 = core_rows.copy()
oldattn129 = core_attn
rng129 = np.random.default_rng(129)
eligible129 = np.flatnonzero(state_df["hour"].to_numpy() > 1)
rows129 = np.sort(rng129.choice(eligible129, 96, replace=False))
core_rows = rows129
hf129 = hflow46[rows129]
ef129 = torch.as_tensor(E_fact95[rows129], dtype=torch.float32)
targeth129 = core_state[rows129,192:960].reshape(-1,4,192)
targeto129 = core_state[rows129,:192]
baseh129 = hf129+ef129

def attn129(x, p, nheads=4, memory=None, ages=None, valid=None,
            age_mode="projected_key"):
    q = core_lin(x, p+".q")
    k, v = core_lin(x if memory is None else memory, p+".kv").chunk(2,-1)
    if memory is not None:
        if age_mode == "direct_key":
            k = k+ages
        elif age_mode == "direct_both":
            k, v = k+ages, v+ages
        elif age_mode in ("projected_key", "projected_both"):
            ka, va = core_lin(ages, p+".kv").chunk(2,-1)
            k = k+ka
            if age_mode == "projected_both":
                v = v+va
    b, nq, width = q.shape
    nk = k.shape[1]
    q = q.reshape(b,nq,nheads,width//nheads).transpose(1,2)
    k = k.reshape(b,nk,nheads,width//nheads).transpose(1,2)
    v = v.reshape(b,nk,nheads,width//nheads).transpose(1,2)
    mask = None if valid is None else valid[:,None,None,:]
    ans = F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    return core_lin(ans.transpose(1,2).reshape(b,nq,width), p+".out")

audit129 = []
try:
    core_attn = attn129
    for include129 in (False, True):
        mem129, secs129, valid129 = _mf121(include129)
        for scale129 in (1.0, 0.1):
            ageinput129 = torch.log1p(secs129)*scale129
            ages129 = (core_lin(ageinput129, "core.age")
                       if "core.age.weight" in W0
                       else mlp0(ageinput129, "core.age"))
            for mode129 in ("projected_key", "projected_both",
                            "direct_key", "direct_both"):
                u129 = baseh129
                for layer129 in ("core.blocks.0.0", "core.blocks.0.1"):
                    u129 = core_layer(u129, layer129, 4,
                                      mem129, ages129, valid129, mode129)
                hn129 = (abar121*hf129
                         +delta121*core_lin(ef129,"core.b_in")
                         +(u129-baseh129))
                hy129, ho129 = coda46(hn129)
                erh129 = float(((hy129-targeth129)**2).mean().sqrt())
                ero129 = float(((ho129-targeto129)**2).mean().sqrt())
                audit129.append((erh129, ero129, include129, scale129, mode129))
                print("include/age-scale/mode:",
                      include129, scale129, mode129,
                      "h RMSE", round(erh129,6),
                      "out RMSE", round(ero129,6))
finally:
    core_attn = oldattn129
    core_rows = oldrows129

print("BEST:", sorted(audit129)[:3])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 129: controlled memory-age reconstruction audit\n"
            "Helper weights are original; NE-only archive composition verified to numerical precision. "
            "Tested current-memory inclusion, age-log scale and direct versus projected age injection "
            "against 96 stored GPU states, without weight changes. Best="+repr(sorted(audit129)[:3])+
            ". Python runs: 129; forwards: 15.\n")
```