Run 101 is the breakthrough. With the memory including the current bundle, and the age embedding added to the memory vectors before the key/value projection, my rebuilt step reproduces the recorded h exactly, to printed precision, on all 57 steps tested. The encoder already matched the z posterior exactly for every bundle type. So I now have an exact local copy of the core.

Next I'll write the confirmed details into the notebook, so they survive a restart, and verify two more things:
- the z posterior over all bundles, using my own encodings rather than the recorded z;
- the initial state for the first bundle.

For the initial state I'll test a few candidates: zeros, or the learned slot vectors, with a zero or very small gap. With those in place I can run whole stays locally, and then the value head, which needs prior samples of z.

```python
with open('/work/notes.md','a') as fh:
    fh.write("""
## Runs 94-101: EXACT REIMPLEMENTATION FOUND (core step matches states to ~1e-6)
- atom vec = code + kind + ekind + kq[0] + [unit, rate_unit, bins, numeric only for kind 0] + numeric@[zv,hv,zd,hd,zr,hr] + numeric.bias.
  unit idx: glucose_finger 0, glucose_lab 5, map_art/cuff 2, insulin 3, norepi 4. rate_unit: insulin 2, norepi 3. bins side 'left'.
- END atoms (kind 2) of drugs go to the WORLD encoder (e), with NO units/attrs. Action encoder gets only kind-0 drug atoms.
- empty set -> encoder applied to empty set (LN(mlp2(size(0)))), both for e and u (NOT zero).
- ctx = out + LN_out(mean_slots(h*exp(-dt*exp(flow.a)))) + gap(log(dt_s)/10); z = z_post([ctx,e,u])[:32]. GELU (erf) everywhere.
- prelude: tokens [hf+slots(4), e, u, z_in(z)], pre-norm layer (attn norm1, ffn norm3), E = prelude_norm(slot outputs).
- blocks: xb = hf+E; per layer: +selfattn(norm1), +cross(norm2(x) -> M + age_emb(log1p(age_s))) [age added to memory BEFORE kv proj; M INCLUDES current e], +ffn(norm3).
- h_new = abar*hf + sp(delta_in)*b_in(E) + (xb-(hf+E)); y = coda(c_out(h_new)) (attn norm1, ffn norm3); out = LN_out(mean(y)).
- attention: kv proj -> k=first 192, v=last 192; 4 heads of 48; scale 1/sqrt(48); no biases.
""")
zerr = defaultdict(list)
for s_ in list(bundles_B.keys())[:30]:
    bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
    for k_ in range(1, len(bl_)):
        e, u, _ = enc_bundle(bl_[k_][1])
        prev = Xt[r0_+k_-1]; dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
        ctx = prev[:192] + ln_(flow_c(prev[192:960].reshape(4,192), dt, 1.0).mean(0),'core.out_norm') + G_in(dt, 's10')
        zh = mlp_(torch.cat([ctx, e, u]), 'z_post', 'gelu')[:32]; zt = Xt[r0_+k_, 960:]
        zerr['drug' if any(a['code'] in ACTION_CODES for a in bl_[k_][1]) else 'other'].append(float(torch.norm(zh-zt)/torch.norm(zt)))
print({k_: (len(v_), float(np.max(v_))) for k_, v_ in zerr.items()})
s_ = list(bundles_B.keys())[0]; bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
e0, u0, _ = enc_bundle(bl_[0][1]); zt0 = Xt[r0_, 960:]; ht0 = Xt[r0_, 192:960].reshape(4,192)
for nm_, h_init in [('zeros', torch.zeros(4,192)), ('slots', T_['core.slots'].clone())]:
    o_init = ln_(h_init.mean(0), 'core.out_norm')
    for dt0 in [1.0, 60.0, 3600.0]:
        ctx = o_init + ln_(flow_c(h_init, dt0, 1.0).mean(0),'core.out_norm') + G_in(dt0, 's10')
        zh = mlp_(torch.cat([ctx, e0, u0]), 'z_post', 'gelu')[:32]
        o_h, h_h = step(o_init, h_init, dt0, e0, u0, zh, e0[None], torch.zeros(1), {'mem_incl_cur': True, 'age_mode': 'add_mem'})
        print(nm_, dt0, 'z err', round(float(torch.norm(zh-zt0)/torch.norm(zt0)),4), 'h err', round(float(torch.norm(h_h-ht0)/torch.norm(ht0)),4))
```