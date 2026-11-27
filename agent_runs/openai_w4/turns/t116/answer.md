The prelude decoder generalizes poorly, so a simple recurrent-matrix edit would mostly inject noise. I’ll examine the attention interface for a dedicated channel that carries the better-validated posterior innovation. The key test is whether one attention head can be repurposed while its original contribution is reconstructed accurately by the others.

```python
import numpy as np, torch

rp_linear_original103 = rp_linear
def capture_attention103(indices):
    captured = {}
    def wrapped(x, prefix):
        if prefix in ("core.prelude.0.attn.q",
                      "core.prelude.0.attn.kv",
                      "core.prelude.0.attn.out"):
            captured[prefix.rsplit(".",1)[-1]] = x.detach().cpu().numpy().copy()
        return rp_linear_original103(x, prefix)
    previous = globals()["rp_linear"]
    globals()["rp_linear"] = wrapped
    try:
        result, detail = capture_transition102(indices)
    finally:
        globals()["rp_linear"] = previous
    return captured, detail

Qparts103, Kparts103, Aparts103 = [], [], []
for start103 in range(0, len(indices102), 192):
    part103, _ = capture_attention103(indices102[start103:start103+192])
    Qparts103.append(part103["q"])
    Kparts103.append(part103["kv"])
    Aparts103.append(part103["out"])
NQ103 = np.concatenate(Qparts103)
NK103 = np.concatenate(Kparts103)
AF103 = np.concatenate(Aparts103)
print("attention input shapes:", NQ103.shape, NK103.shape, AF103.shape)
print("query/key input difference RMS:", float(np.sqrt(np.mean((NQ103-NK103)**2))))
print("config:", VOCAB["config"])
print("attention constants:", attention67.__code__.co_consts)

q103 = rp_linear_original103(torch.tensor(NQ103[:24]), "core.prelude.0.attn.q")
kv103 = rp_linear_original103(torch.tensor(NK103[:24]), "core.prelude.0.attn.kv")
k103, v103 = kv103.chunk(2, dim=-1)
head_checks103 = []
for nh103 in (3,4,6,8,12,16):
    dh103 = 192//nh103
    qq103 = q103.reshape(24,7,nh103,dh103).transpose(1,2)
    kk103 = k103.reshape(24,7,nh103,dh103).transpose(1,2)
    vv103 = v103.reshape(24,7,nh103,dh103).transpose(1,2)
    pp103 = torch.softmax(qq103 @ kk103.transpose(-1,-2) / np.sqrt(dh103), dim=-1)
    aa103 = (pp103@vv103).transpose(1,2).reshape(24,7,192).numpy()
    err103 = float(np.sqrt(np.mean((aa103-AF103[:24])**2)))
    head_checks103.append((err103, nh103))
print("head-count reconstruction:", head_checks103)
heads103 = min(head_checks103)[1]
assert min(head_checks103)[0] < 1e-5
width103 = 192//heads103

print("normalization / recurrent weights:")
for n103 in NAMES:
    if n103.startswith(("core.prelude_norm", "core.b_in", "core.prelude.0.norm1")):
        print(n103, SHAPES[n103])
gamma103 = np.asarray(load_original("core.prelude.0.norm1.weight"))
bias103 = np.asarray(load_original("core.prelude.0.norm1.bias"))
print("constant-normalized-token denominator:", float(np.sum(bias103/gamma103)))

slots103 = AF103[:,:4].reshape(-1,192).astype(np.float64)
out_weight103 = np.asarray(load_original("core.prelude.0.attn.out.weight"))
target_attention103 = slots103 @ out_weight103.T
trslots103 = np.repeat(train102,4)
teslots103 = np.repeat(test102,4)
head_comp103 = {}
head_table103 = []
for head103 in range(heads103):
    keep103 = np.setdiff1d(np.arange(192), np.arange(head103*width103,(head103+1)*width103))
    xx103 = np.column_stack((slots103[:,keep103], np.ones(len(slots103))))
    gg103 = xx103[trslots103].T@xx103[trslots103]
    rr103 = xx103[trslots103].T@target_attention103[trslots103]
    reg103 = np.eye(len(keep103)+1)*1e-5
    reg103[-1,-1] = 1e-9
    cc103 = np.linalg.solve(gg103+reg103, rr103)
    ee103 = xx103@cc103-target_attention103
    head_comp103[head103] = (keep103,cc103)
    head_table103.append((head103,
                         float(np.sqrt(np.mean(ee103[teslots103]**2))),
                         float(np.sqrt(np.mean(ee103[trslots103]**2)))))
print("head removal held-out/train output RMS:", head_table103)
print("original attention-output RMS:", float(np.sqrt(np.mean(target_attention103**2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 103\nPrelude innovation decoding failed (held-out insulin R2 only ~0.05–0.10), rejecting the simple B_in shortcut. Captured exact attention inputs and tested whether a redundant head can be compensated before creating an explicit gated-innovation channel. No edits.\n")
```
