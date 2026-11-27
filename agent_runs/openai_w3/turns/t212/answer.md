The reserved attention head has two unused channels and a reliable constant channel, which allows a centered dose signal to be added through weights alone. I’ll first compensate for the latent-channel changes without a dose payload, then install a pilot if held-out reconstruction is accurate. The factual forward test will decide whether that local preservation holds in the model.

```python
import time
t192 = time.time()
base192 = {k: tensor153(v).clone() for k, v in zero169.items()}
orig192 = {k: load_original(k) for k in zero169}
tokens192 = tensor153(tokens_original155).clone()
z192 = posterior153(C154, E154, U154, post187)
tokens192[:, 6] = F.linear(z192, tensor153(zin154), tensor153(zb154))

r_orig192 = prelude155(tokens_original155, overrides=orig192)
r_new192 = prelude155(tokens192, overrides=base192)
def hidden192(after):
    return F.gelu(linear(layer_norm(after, "core.prelude.0.norm3"),
                         "core.prelude.0.ffn.0"))
ho192 = hidden192(r_orig192[1])
hn192 = hidden192(r_new192[1])
raw_target192 = r_orig192[1] + F.linear(
    ho192, orig192["core.prelude.0.ffn.2.weight"],
    orig192["core.prelude.0.ffn.2.bias"])
raw_base192 = r_new192[1] + F.linear(
    hn192, base192["core.prelude.0.ffn.2.weight"],
    base192["core.prelude.0.ffn.2.bias"])

H192 = arr183(hn192).reshape(-1, hn192.shape[-1]).astype(np.float64)
R192 = arr183(raw_target192-raw_base192).reshape(-1,192).astype(np.float64)
tr192 = np.repeat(train154,4)
va192 = np.repeat(valid154,4)
hm192, rm192 = H192[tr192].mean(0), R192[tr192].mean(0)
hc192, rc192 = H192[tr192]-hm192, R192[tr192]-rm192
cov192 = hc192.T @ hc192 / len(hc192)
rhs192 = hc192.T @ rc192 / len(hc192)
eig192, vec192 = np.linalg.eigh(cov192)
scale192 = np.trace(cov192)/len(cov192)
best192 = None
print("=== NO-PAYLOAD FACTUAL COMPENSATION ===")
for ridge192 in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    dw192 = vec192 @ ((vec192.T @ rhs192) /
                     (eig192[:,None] + ridge192*scale192 + 1e-10))
    db192 = rm192-hm192@dw192
    raw192 = raw_base192 + tensor153(
        (H192@dw192+db192).reshape(n154,4,192))
    ep192 = layer_norm(raw192, "core.prelude_norm")
    err192 = arr183(ep192-r_orig192[0])
    score192 = float(np.sqrt(np.mean(err192[valid154]**2)))
    print(ridge192, "E train/holdout RMS",
          np.sqrt(np.mean(err192[train154]**2)), score192)
    if best192 is None or score192 < best192[0]:
        best192 = (score192, ridge192, dw192.copy(), db192.copy())
base192["core.prelude.0.ffn.2.weight"] += tensor153(best192[2].T)
base192["core.prelude.0.ffn.2.bias"] += tensor153(best192[3])
base192.update({k: tensor153(v).clone() for k,v in post187.items()})
base192["z_in.weight"] = tensor153(zin154)
base192["z_in.bias"] = tensor153(zb154)

pilot192 = {k:v.clone() for k,v in base192.items()}
decoder192 = decoders187[0.1]
centred192 = decoder192 - vconst191[:,None] * (mu187@decoder192)[None,:]
for d192 in range(2):
    pilot192["core.prelude.0.attn.kv.weight"][192+144+d192] = tensor153(centred192[:,d192])
    pilot192["core.prelude.0.attn.out.weight"][:,144+d192] = tensor153(directions187[:,d192])
signal192 = x0_187@centred192
print("Centered factual signal RMS:", np.sqrt(np.mean(signal192**2,axis=0)))
print("Target/presence ranges:", Y154.min(0), Y154.max(0))
p192 = prelude155(tokens192, overrides=pilot192)
print("Pilot factual E RMS:", np.sqrt(np.mean(arr183(p192[0]-r_orig192[0])**2)))
print("Selected compensation:", best192[:2])
install192 = bool(np.isfinite(best192[0]) and best192[0] < 0.005)
if install192:
    for k,v in pilot192.items():
        save_edit(k,v)
print("Installed:", install192, "edits:", edited(), "seconds:", time.time()-t192)
with open("/work/notes.md", "a") as f:
    f.write(f"\nPython run 192; forwards 10. Built compensated latent transport with low-penalty paired-contrast dose decoder and primary physiological directions. No-payload whole-stay-held-out E RMS={best192[0]:.8g}; pilot installed={install192}. This is a reversible pilot, not validated repair; factual NLL and actual rollouts must decide acceptance.\n")
```
