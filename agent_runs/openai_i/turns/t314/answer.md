The refit stopped because I used the 258-context probe indices to label the larger compensation sample. No weights changed, and the head comparison remains untested. I’ll recover the full sample’s stay labels and resume the regressions. Lower stay-held error will justify a GPU test, but will not by itself establish factual fit.

```python
# Recover full-sample metadata rather than indexing through the 258-row probe.
nfull276 = len(ot275)
sid_full276 = None
metadata276 = []
for k, v in list(globals().items()):
    if "stay" in k.lower() and hasattr(v, "shape") and tuple(v.shape) == (nfull276,):
        a = arr275(v)
        if len(np.unique(a)) >= 20:
            metadata276.append((k, a))
if metadata276:
    metadata276.sort(key=lambda x: ("all" not in x[0].lower(), x[0]))
    source276, sid_full276 = metadata276[0]
else:
    for p in sorted(Path("/work").glob("fwd_*/states_original.npz")):
        with np.load(p) as d:
            sk = [k for k in d.files if "stay" in k.lower()
                  and d[k].shape == (nfull276,)]
            zk = [k for k in d.files if d[k].ndim == 2
                  and d[k].shape[0] == nfull276 and d[k].shape[1] in (32, 992)]
            for k in zk:
                zcheck = d[k][:, -32:]
                if np.sqrt(np.mean((zcheck - arr275(zorig222))**2)) < .01 and sk:
                    sid_full276 = d[sk[0]].copy()
                    source276 = str(p) + ":" + sk[0]
                    break
        if sid_full276 is not None:
            break
assert sid_full276 is not None, (
    "No verified full metadata; available state files: "
    + str(list(Path('/work').glob('fwd_*/states_original.npz')))
)
sid275 = sid_full276[ix275]
uniq275 = np.unique(sid275)
hold275 = np.isin(sid275, uniq275[::5])
train275 = np.repeat(~hold275, 4)
test275 = ~train275
print("Metadata:", source276, "bundles:", len(ix275),
      "stays:", len(uniq275), "held stays:", len(np.unique(sid275[hold275])))

oq275 = np.asarray(load_original("core.prelude.0.attn.q.weight"))
okv275 = np.asarray(load_original("core.prelude.0.attn.kv.weight"))
oout275 = np.asarray(load_original("core.prelude.0.attn.out.weight"))
ofw275 = np.asarray(load_original("core.prelude.0.ffn.2.weight"))
ofb275 = np.asarray(load_original("core.prelude.0.ffn.2.bias"))
slots275 = arr275(orig_tokens222[:, :4]).reshape(-1, 192)

def ff275(x, fw=ofw275, fb=ofb275):
    t = ten275(x)
    h = F.gelu(F.linear(F.layer_norm(
        t, (192,), orig220["core.prelude.0.norm3.weight"],
        orig220["core.prelude.0.norm3.bias"]),
        orig220["core.prelude.0.ffn.0.weight"],
        orig220["core.prelude.0.ffn.0.bias"]))
    y = t + F.linear(h, ten275(fw), ten275(fb))
    return arr275(h).astype(np.float64), arr275(y).astype(np.float64)

fo275 = arr275(att_features220(orig_tokens222, oq275, okv275)).reshape(-1, 192)
oa275 = fo275 @ oout275.T
_, target275 = ff275(slots275 + oa275)

def ridge275(features, target, lambdas):
    a, y = np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64)
    m, s = a[train275].mean(0), a[train275].std(0) + 1e-4
    ym = y[train275].mean(0)
    ax = (a - m) / s
    g = ax[train275].T @ ax[train275]
    b = ax[train275].T @ (y[train275] - ym)
    best = None
    for lam in lambdas:
        co = solve(g + np.eye(g.shape[0]) * (train275.sum() * lam),
                   b, assume_a="pos", check_finite=False)
        coef = co / s[:, None]
        bias = ym - m @ coef
        err = float(np.sqrt(np.mean((a[test275] @ coef + bias - y[test275]) ** 2)))
        if best is None or err < best[0]:
            best = (err, lam, coef, bias)
    return best

baseout275 = c271["core.prelude.0.attn.out.weight"].copy()
baseout275[:, 48:50] = 0
basefeat275 = arr275(att_features220(
    tok275, c271["core.prelude.0.attn.q.weight"],
    c271["core.prelude.0.attn.kv.weight"])).reshape(-1, 192)
_, basey275 = ff275(slots275 + basefeat275 @ baseout275.T,
                    c271["core.prelude.0.ffn.2.weight"],
                    c271["core.prelude.0.ffn.2.bias"])
baseerr275 = float(np.sqrt(np.mean((basey275[test275] - target275[test275]) ** 2)))
print("Current corrected zero-response held RMS:", baseerr275)

candidates275, scores275 = {}, {}
for head in range(4):
    start = head * 48
    q, kv = oq275.copy(), okv275.copy()
    q[start:start+48] = c271["core.prelude.0.attn.q.weight"][48:96]
    kv[start:start+48] = c271["core.prelude.0.attn.kv.weight"][48:96]
    kv[192+start:192+start+3] = c271["core.prelude.0.attn.kv.weight"][240:243]
    features = arr275(att_features220(tok275, q, kv)).reshape(-1, 192)
    keep = np.setdiff1d(np.arange(192), np.arange(start, start+3))
    const = features[:, start+2]
    assert np.std(const) < 1e-4 and abs(np.mean(const)) > 1e-5
    ae, al, ac, ab = ridge275(features[:, keep], oa275,
                              [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    out = np.zeros_like(oout275)
    out[:, keep] = ac.T.astype(np.float32)
    out[:, start+2] = (ab / np.mean(const)).astype(np.float32)
    hidden, y = ff275(slots275 + features @ out.T)
    fe, fl, fc, fb = ridge275(hidden, target275 - y,
                              [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    candidate = {k: v.copy() for k, v in c271.items()}
    candidate["core.prelude.0.attn.q.weight"] = q
    candidate["core.prelude.0.attn.kv.weight"] = kv
    candidate["core.prelude.0.attn.out.weight"] = out
    candidate["core.prelude.0.ffn.2.weight"] = (ofw275 + fc.T).astype(np.float32)
    candidate["core.prelude.0.ffn.2.bias"] = (ofb275 + fb).astype(np.float32)
    candidate["core.prelude.0.attn.out.weight"][:, start:start+2] = \
        c271["core.prelude.0.attn.out.weight"][:, 48:50]
    candidates275[head], scores275[head] = candidate, (fe, ae, fl, al)
    np.savez(f"/work/head275_{head}.npz", **candidate)
    print("head", head, "held final RMS", round(fe, 6),
          "attention RMS", round(ae, 6), "FFN ridge", fl, "attention ridge", al)

besthead275 = min(scores275, key=lambda h: scores275[h][0])
installed275 = scores275[besthead275][0] < baseerr275
if installed275:
    for k in list(edited()):
        drop_edit(k)
    for k, v in candidates275[besthead275].items():
        save_edit(k, v.astype(np.float32))
print("Best head:", besthead275, "provisionally installed:", installed275)
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 276 — corrected-input head comparison\n"
            f"- Full stay metadata recovered from {source276}.\n"
            f"- Baseline held RMS {baseerr275:.6f}; candidate scores {scores275}.\n"
            f"- Best head {besthead275}; provisionally installed {installed275}.\n"
            "- Closed-form regressions only; response directions retained at gain 1.\n"
            "- No candidate is accepted without GPU factual validation.\n")
```