The dose is in the state after all. Right after an insulin bundle, the random part of that bundle's dose can be decoded with cross-validated R² of about 0.42–0.44, from out, from the slot mean or from all of h. The latent z gives 0.33.

My earlier probe of q failed because that rebuild used ReLU in the gap and prior networks, which corrupted q. Since out enters the head's view directly, the information should also be readable from a correct q.

That makes a concrete plan possible. In the glucose value head, I would turn pairs of never-active hidden units into gated linear units. They would read a probe of q for "dose minus the dose the readings imply", scaled to make up the missing effect. A rough estimate puts the factual-NLL cost at about 0.0015 nats or less, within the 0.005 limit. Norepinephrine on MAP would be handled the same way.

First I need the head rebuild right under GELU. This run re-screens, on stay 1010001440:
- the time flow on or off;
- the gap input;
- the type_emb row.

If the head error falls well below the 2.6 units of the last check, I can trust q-based probes and the search for unused units.

```python
def head_pred(typ=1, flow=True, gmode='log(s)/10', gval=None, feats_fn=FF['B'], ret_q=False):
    out_p, qs = [], []
    for kprev, dt, ev, lv, la, pm_ in rows_:
        x_ = Xs[kprev]; out = x_[:192]; h = x_[192:960].reshape(4, 192)
        hf = h * np.exp(-dt * flow_k) if flow else h
        gin = gval if gval is not None else GF[gmode](dt)
        ctx = out + ln(hf.mean(0), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.array([gin]), 'gap')
        zp = mlp(ctx, 'z_prior'); view = ctx + lin(zp[:32], 'z_head')
        q = ln(view + W['type_emb.weight'][typ], W['query_norm.weight'], W['query_norm.bias']); qs.append(q)
        c = CODE[ev]; has = 0.0 if np.isnan(lv) else 1.0
        zl = 0.0 if not has else zsc(ev, 'value', lv); lb = 0 if not has else fbin(ev, 'value', lv)
        ag = 0.0 if not has else np.log1p(la / 3600.0) / 10
        wc = q + W['embed.code.weight'][c] + lin(np.array(feats_fn(has, ag, zl), float), 'last_numeric') + W['embed.bins.value.weight'][lb]
        lg = mlp(wc, 'attr_heads.value_bin') + W['embed_transition.value_bin.weight'][c * 17 + lb]
        p = np.exp(lg - lg.max()); p /= p.sum(); out_p.append(np.sum(p[1:] * CEN[ev]))
    return (np.array(out_p), np.array(qs)) if ret_q else np.array(out_p)
res_ = []
for typ in range(15):
    for flow in [True, False]:
        for gmode in ['log(s)/10', 'log1p(h)']:
            pr = head_pred(typ, flow, gmode); res_.append((np.mean(np.abs(pr - pm)), typ, flow, gmode))
res_.sort(); print(res_[:6])
b_ = res_[0]; report(head_pred(b_[1], b_[2], b_[3]), 'best')
```