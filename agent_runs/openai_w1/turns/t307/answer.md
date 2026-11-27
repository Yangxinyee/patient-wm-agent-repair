The update equations show where to measure the remaining feedback: after the coda, through the next prelude. I’ll compare the proposed injection directions there and recover the identification notes. Small transformed feedback would support a cautious factual test; large feedback would require a different injection direction.

```python
import torch.nn.functional as F
from pathlib import Path

print("=== RECORD-BASED IDENTIFICATION ===")
n269 = Path("/work/notes.md").read_text()
print(n269[:3200])
p269 = n269.rfind("## Run 236")
print("\nSAFE236:", n269[p269:p269+1100] if p269 >= 0 else "heading absent")

def prelude269(tokens):
    p = "core.prelude.0"
    t = tokens + rx_attn(rx_norm(tokens, p+".norm1"), p+".attn")
    t = t + rx_mlp(rx_norm(t, p+".norm3"), p+".ffn", "gelu")
    return rx_norm(t[:, :4], "core.prelude_norm")

E269 = prelude269(rx_tokens126)
x269 = rx_hflow + E269
u269 = x269
for p in ("core.blocks.0.0", "core.blocks.0.1"):
    u269 = rx_layer2(u269, p, "gelu", memory=rx_mem, age=rx_ages,
                     valid=rx_memvalid, mode="before_k", age_act="gelu")
d269 = F.softplus(rx_W["core.delta_in"])
a269 = torch.exp(-d269 * torch.exp(rx_W["core.a_in"]))
h269 = a269 * rx_hflow + d269 * _rxlinear257(E269, "core.b_in") + u269 - x269

def coda269(v):
    v = torch.as_tensor(v, dtype=h269.dtype)
    y = rx_layer2(_rxlinear257(h269 + d269*v, "core.c_out"), "core.coda.0", "gelu")
    return y, rx_norm(y.mean(1), "core.out_norm")

y269, out269 = coda269(np.zeros(192, dtype=np.float32))
print("\n=== LOCAL RECONSTRUCTION ===")
echeck269, ycheck269, ocheck269 = transition214(None, "gelu")
print("E/y/out max error", *[float((a-b).abs().max()) for a,b in
      ((E269,echeck269),(y269,ycheck269),(out269,ocheck269))])

flow269 = torch.exp(-torch.as_tensor(rx_dt, dtype=torch.float32).reshape(-1,1,1)
                     * torch.exp(rx_W["core.flow.a"]))
for pname, P269 in (("unconstrained", fits263[1.0]), ("immediate-null", fits266[1.0])):
    cols269 = []
    for j in range(2):
        yp, _ = coda269(0.03*B257[:,j])
        ym, _ = coda269(-0.03*B257[:,j])
        dy = (yp-ym)/0.06
        tp = rx_tokens126.clone()
        tm = rx_tokens126.clone()
        tp[:,:4] += 0.03*dy*flow269
        tm[:,:4] -= 0.03*dy*flow269
        de = ((prelude269(tp)-prelude269(tm))/0.06).numpy()
        cols269.append(de @ P269)
    feedback269 = np.stack(cols269, axis=-1)
    print(pname, "mean feedback", np.round(feedback269.mean((0,1)),3),
          "p95 abs", np.round(np.quantile(np.abs(feedback269),.95,axis=(0,1)),3))

print("\n=== POSTERIOR INPUT ALIGNMENT ===")
for name in ("ds_in", "ds_t"):
    a = np.asarray(globals()[name])[rx_pick]
    print(name, "e/u RMS",
          float(np.sqrt(np.mean((a[:,192:384]-rx_tokens126[:,4].numpy())**2))),
          float(np.sqrt(np.mean((a[:,384:576]-rx_tokens126[:,5].numpy())**2))))
print("ctx objects:", [(n, tuple(globals()[n].shape)) for n in list(globals())
                       if "ctx" in n and isinstance(globals()[n], (np.ndarray,torch.Tensor))][:15])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 269\nReconstructed prelude, recurrent injection and coda locally; measured transformed slot-feedback of existing dose projections. Recovered early identification and posterior-input alignment. Safe236 unchanged. Counts: 269 Python runs, 28 forwards.\n")
```
