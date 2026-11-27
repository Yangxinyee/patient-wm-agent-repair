The slot queries in the prelude attend almost entirely to the `u` token at insulin bundles: heads 1 and 2 put exactly 1.0 on it, and heads 0 and 3 put 0.96 on it. So every head's output at a drug bundle is essentially a linear function of `norm1(u)`. That is an ideal place for a rank-1, key→value edit: the key is the dose representation, the value is the desired state change.

The dose barely gets through today for a scale reason. The slot state is huge (‖y‖ ≈ 1792), so doubling the dose moves y by only ~0.5% and moves `out` by 0.04 out of ~14. The `u` contribution is drowned out at the normalisations downstream.

Before designing the edit I need four things:
- how large the prelude residual is (its std sets how much a write is diluted by `prelude_norm`), and how much that varies;
- the attention pattern at glucose-reading bundles (where heads 1–2 go there decides interference);
- the gate values (`abar`, `sp(delta_in)`, flow decay per hour), which decide how a written signal persists;
- the true policy mean rate per glucose band, and whether the fine-bin edges line up with the policy thresholds, which matters for centring the dose signal so factual fit doesn't degrade.

```python
sig=[]; ao=[]; ffo=[]; hfn=[]; Ar=[]; tread=[]
for sid in list(bundles_B.keys())[:30]:
    R = run_tf(sid)
    for r in R:
        E, att, x, tok = prelude_full(r['hf'], r['e'], r['u'], r['z'])
        P='core.prelude.0.'
        a = LNt(tok, P+'norm1'); kv = a @ W[P+'attn.kv.weight'].T
        vh = kv[:,192:].view(7,4,48).transpose(0,1)
        o = (att @ vh).transpose(0,1).reshape(4,192) @ W[P+'attn.out.weight'].T
        isd = any(a_['code'] in ('insulin','norepinephrine') and a_['kind']==0 for a_ in r['ats'])
        isg = any(a_['code']=='glucose_finger' for a_ in r['ats'])
        if isd:
            xpre = tok[:4]+o
            sig.append(float(x.std(-1, unbiased=False).mean())); ao.append(float(o.norm(dim=-1).mean()))
            ffo.append(float((x-xpre).norm(dim=-1).mean())); hfn.append(float(r['hf'].norm(dim=-1).mean()))
        elif isg:
            Ar.append(att.mean(1).numpy())
sig=np.array(sig); Ar=np.array(Ar)
print("drug bundles: sigma(x) mean %.1f sd %.1f min %.1f max %.1f | |attn_out| %.1f | |ffn| %.1f | |hf| %.1f" % (sig.mean(), sig.std(), sig.min(), sig.max(), np.mean(ao), np.mean(ffo), np.mean(hfn)))
print("reading bundles attention per head (cols: slot0-3,e,u,z):")
for hh in range(4): print("  head", hh, Ar[:,hh,:].mean(0).round(3))
ab = abar.numpy() if hasattr(abar,'numpy') else np.asarray(abar)
spv = sp.numpy() if hasattr(sp,'numpy') else np.asarray(sp)
fl = np.exp(-3600*np.exp(W['core.flow.a'].numpy()))
print("abar quantiles", np.quantile(ab,[0,.1,.5,.9,1]).round(3), "| sp(delta_in)", np.quantile(spv,[0,.1,.5,.9,1]).round(3))
print("flow decay per hour quantiles", np.quantile(fl,[0,.1,.5,.9,1]).round(3))
ge = np.array(VOCAB['bins'] if 'bins' in VOCAB else [])
print("VOCAB keys:", list(VOCAB.keys()))
```