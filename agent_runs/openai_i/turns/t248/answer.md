The latent token was selected almost perfectly on held-out bundles without an anchor. I’ll use that selector and add two small, isolated innovation signals, keeping the other three attention heads unchanged. Then I’ll fit neutral reconstruction weights. Low held-out error and accurate innovation transport would support a GPU factual test.

```python
# Small-signal routing: no large anchor, two isolated innovation coordinates.
route222 = {k: np.array(v, copy=True) for k, v in neutral209.items()}
gate222 = routes221[0]["gate"].copy()
zorig222 = F.linear(
    F.gelu(F.linear(input196, t220(load_original("z_post.0.weight")),
                    t220(load_original("z_post.0.bias")))),
    t220(load_original("z_post.2.weight")),
    t220(load_original("z_post.2.bias")))[:, :32]
orig_token222 = F.linear(zorig222, t220(load_original("z_in.weight")),
                        t220(load_original("z_in.bias")))
orig_tokens222 = old_tokens196.clone()
orig_tokens222[:, 6] = orig_token222[source201]
print("Recomputed original versus cached token RMS:",
      rms220((orig_tokens222[:, 6] - old_tokens196[:, 6]).numpy()))

baseW222 = np.asarray(route222["z_in.weight"], dtype=np.float64)
baseb222 = np.asarray(route222["z_in.bias"], dtype=np.float64)
constraints222 = np.column_stack(
    [baseW222, baseb222, np.ones(192), gamma221 * gate222])
u222, sv222, _ = np.linalg.svd(constraints222, full_matrices=False)
basis222 = u222[:, sv222 > sv222.max() * 1e-6]
axes222 = []
for seed222 in (np.asarray(axis_ins196), np.asarray(axis_ne196)):
    v222 = seed222.astype(np.float64).copy()
    v222 -= basis222 @ (basis222.T @ v222)
    for a222 in axes222:
        v222 -= a222 * (a222 @ v222)
    v222 /= np.linalg.norm(v222)
    axes222.append(v222)
axes222 = np.stack(axes222, axis=1)
signal_scale222 = 0.01 * radius221
private_cols222 = np.asarray(private_indices209, dtype=int)
route222["z_in.weight"][:, private_cols222] = (
    signal_scale222 * axes222).astype(np.float32)
token222 = F.linear(z221, t220(route222["z_in.weight"]),
                    t220(route222["z_in.bias"]))
tokens222 = old_tokens196.clone()
tokens222[:, 6] = token222[source201]

q222 = orig220["core.prelude.0.attn.q.weight"].numpy().copy()
kv222 = orig220["core.prelude.0.attn.kv.weight"].numpy().copy()
hs222 = 48 * int(private_head196)
constant222 = (1.0 / gamma221) / constant_sum221
sigma222 = float(torch.median(
    torch.sqrt(((base_token221 - base_token221.mean(-1, keepdim=True))**2).mean(-1)
               + 1e-5)))
q222[hs222:hs222+48] = 0
q222[hs222] = constant222
kv222[hs222:hs222+48] = 0
kv222[hs222] = np.sqrt(48.) * 20. * gate222
for j222 in range(2):
    vr222 = sigma222 / signal_scale222 * axes222[:, j222] / gamma221
    vr222 -= float(beta221 @ vr222) * constant222
    kv222[192 + hs222 + j222] = vr222.astype(np.float32)
kv222[192 + hs222 + 2] = constant222.astype(np.float32)

fo222, fn222 = [], []
for start222 in range(0, len(tokens222), 256):
    sl222 = slice(start222, start222 + 256)
    fo222.append(att_features220(
        orig_tokens222[sl222],
        orig220["core.prelude.0.attn.q.weight"],
        orig220["core.prelude.0.attn.kv.weight"]))
    fn222.append(att_features220(tokens222[sl222], q222, kv222))
fo222 = torch.cat(fo222).reshape(-1, 192).numpy().astype(np.float64)
fn222 = torch.cat(fn222).reshape(-1, 192).numpy().astype(np.float64)
target222 = fo222 @ orig220["core.prelude.0.attn.out.weight"].numpy().T
outbase222 = orig220["core.prelude.0.attn.out.weight"].numpy().copy()
outbase222[:, hs222:hs222+2] = 0
r222 = target222 - fn222 @ outbase222.T

# Closed-form residual regression; use the donor's constant-value column as bias.
constcol222 = hs222 + 2
keep222 = np.array([j for j in range(192)
                   if j not in (hs222, hs222+1, constcol222)])
m222 = fn222[fit220][:, keep222].mean(0)
s222 = np.maximum(fn222[fit220][:, keep222].std(0), 0.02)
a222 = np.column_stack([(fn222[:, keep222] - m222) / s222,
                       np.ones(len(fn222))])
g222 = a222[fit220].T @ a222[fit220] / fit220.sum()
c222 = a222[fit220].T @ r222[fit220] / fit220.sum()
fits222 = []
for lam222 in (1e-5, 1e-4, 1e-3, 0.01, 0.1):
    penalty222 = np.eye(a222.shape[1]) * lam222
    penalty222[-1, -1] = 1e-10
    co222 = sla.solve(g222 + penalty222, c222, assume_a="pos",
                     check_finite=False)
    er222 = r222 - a222 @ co222
    sc222 = rms220(er222[~fit220])
    fits222.append((sc222, lam222, co222))
    print("Attention ridge", lam222, "train RMS", rms220(er222[fit220]),
          "held RMS", sc222)
bestatt222 = min(fits222, key=lambda x: x[0])
out222 = outbase222.astype(np.float64)
out222[:, keep222] += (bestatt222[2][:-1] / s222[:, None]).T
out222[:, constcol222] += (
    bestatt222[2][-1] - m222 @ (bestatt222[2][:-1] / s222[:, None]))
route222["core.prelude.0.attn.q.weight"] = q222
route222["core.prelude.0.attn.kv.weight"] = kv222
route222["core.prelude.0.attn.out.weight"] = out222.astype(np.float32)

xo222 = orig_tokens222[:, :4].reshape(-1, 192) + t220(target222)
xn222 = tokens222[:, :4].reshape(-1, 192) + t220(fn222 @ out222.T)
yo222 = xo222 + F.linear(ffn_features220(xo222),
                        orig220["core.prelude.0.ffn.2.weight"],
                        orig220["core.prelude.0.ffn.2.bias"])
hn222 = ffn_features220(xn222)
yn222 = xn222 + F.linear(hn222, orig220["core.prelude.0.ffn.2.weight"],
                        orig220["core.prelude.0.ffn.2.bias"])
rf222 = (yo222 - yn222).numpy().astype(np.float64)
af222 = hn222.numpy().astype(np.float64)
mf222 = af222[fit220].mean(0)
sf222 = np.maximum(af222[fit220].std(0), 0.05)
acf222 = (af222 - mf222) / sf222
rmf222 = rf222[fit220].mean(0)
gf222 = acf222[fit220].T @ acf222[fit220] / fit220.sum()
cf222 = acf222[fit220].T @ (rf222[fit220] - rmf222) / fit220.sum()
fitsff222 = []
print("Uncompensated final-slot held RMS:", rms220(rf222[~fit220]))
for lam222 in (1e-4, 1e-3, 0.01, 0.1, 1.0):
    co222 = sla.solve(gf222 + lam222*np.eye(768), cf222,
                     assume_a="pos", check_finite=False)
    er222 = rf222 - (acf222 @ co222 + rmf222)
    sc222 = rms220(er222[~fit220])
    fitsff222.append((sc222, lam222, co222))
    print("FFN ridge", lam222, "train RMS", rms220(er222[fit220]),
          "held RMS", sc222)
bestff222 = min(fitsff222, key=lambda x: x[0])
dw222 = (bestff222[2] / sf222[:, None]).T
db222 = rmf222 - mf222 @ dw222.T
route222["core.prelude.0.ffn.2.weight"] = (
    orig220["core.prelude.0.ffn.2.weight"].numpy() + dw222).astype(np.float32)
route222["core.prelude.0.ffn.2.bias"] = (
    orig220["core.prelude.0.ffn.2.bias"].numpy() + db222).astype(np.float32)
np.savez("/work/small_route222.npz", **route222)
for j222 in range(2):
    v222 = z221[source201, int(private_cols222[j222])].numpy()
    tr222 = fn222.reshape(-1, 4, 192)[:, :, hs222+j222].mean(1)
    active222 = np.abs(v222) > 1e-4
    slope222 = float(v222[active222] @ tr222[active222] /
                     (v222[active222] @ v222[active222]))
    print("Innovation", j222, "transport slope", slope222,
          "inactive RMS", rms220(tr222[~active222]))
print("Saved neutral small route; no new weights installed.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 222 — small-signal, anchor-free routing candidate\n"
            f"- Built isolated innovation axes; latent-token selector uses no anchor.\n"
            f"- Best held attention RMS {bestatt222[0]:.6g}; "
            f"best held final-slot RMS {bestff222[0]:.6g}.\n"
            "- Saved small_route222.npz with zero response output columns. Not installed yet; requires factual validation.\n")
```