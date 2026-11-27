The reconstructed head matches the tool to within the sampling noise floor: MAE 1.32 against an expected 1.28 for 4-draw versus 8-draw predictions. No extra feature column helps. So the head is identified:

- the measurement type index is 1;
- `last_numeric` takes has-last in column 0, log1p(age in seconds)/10 in column 3 and last z in column 6;
- gap uses log(dt)/10, z comes from 4 prior draws, and the activation is GELU.

This means I can compute factual value NLL offline for any edit confined to the value head. Next I will record this in the notebook and save the working head code to /work/lib.py so that a restart does not lose it again.

```python
lib2 = r'''
# ---- validated value-head reconstruction (run 72-73): MAE vs tool 1.32 = noise floor of 4 z-draws
# type_idx for measurements = 1; last_numeric features: col0 has_last, col3 log1p(age_s)/10, col6 z_last_value (clipped), others 0
# gap input log(dt_s)/10; z from prior (first 32 = mu, last 32 = log sd); GELU everywhere; predicted mean = sum_{b>=1} p_b*centre[b-1]
def q_from_rows(Xs, prev_idx, dt_s, nz=8, seed=0, type_idx=1):
    gen = torch.Generator().manual_seed(seed)
    xs = Xs[prev_idx]; out = xs[:, :192]; h4 = xs[:, 192:960].reshape(-1,4,192)
    ctx = ctx_from_state(out, h4, dt_s)
    zs = z_draws(ctx, nz, gen)
    view = ctx[None] + lin(zs, 'z_head')
    return ln(view + T('type_emb.weight')[type_idx], 'query_norm')
def reg_feats(has, age_s, zl):
    f = torch.zeros(len(has), 8, dtype=torch.float64)
    f[:,0] = has; f[:,3] = torch.log1p(age_s)/10; f[:,6] = zl
    return f
def value_logits(Q, code, lvb, f, W0=None, b0=None, W2=None, b2=None):
    W0 = T('attr_heads.value_bin.0.weight') if W0 is None else W0; b0 = T('attr_heads.value_bin.0.bias') if b0 is None else b0
    W2 = T('attr_heads.value_bin.2.weight') if W2 is None else W2; b2 = T('attr_heads.value_bin.2.bias') if b2 is None else b2
    const = T('embed.code.weight')[code] + T('embed.bins.value.weight')[lvb] + f @ T('last_numeric.weight').T + T('last_numeric.bias')
    wc = Q + const[None]
    hid = torch.nn.functional.gelu(wc @ W0.T + b0)
    return hid @ W2.T + b2 + T('embed_transition.value_bin.weight')[code*17 + lvb][None], wc, hid
'''
open('/work/lib.py','a').write(lib2); exec(lib2)
with open('/work/notes.md','a') as fh:
    fh.write('''
## Runs 63-73 (after restart 2): value head reconstructed exactly (code in /work/lib.py)
- Codes: mc 5, gf 6, ma 7, ins 8, ne 9, gl 10. Value head = MLP 192-192(GELU)-17 shared across codes + table embed_transition.value_bin[code*17+last bin].
- with_code = query_norm(view + type_emb[1]) + embed.code[c] + embed.bins.value[last bin] + last_numeric(f) ; f: col0 has_last, col3 log1p(age_s)/10, col6 z_last (clip +-10).
- view = out + out_norm(mean_slots(h*exp(-dt*exp(flow.a)))) + gap(log(dt_s)/10) + z_head(z), z ~ prior (mu=first 32, logsd=last 32), 4 draws averaged.
- Local predicted means match tool within noise floor (MAE 1.32 vs floor 1.28) on 38 stays overlapping sel1 (states fwd_2) and fwd_3 (factual).
- Glucose centres 73..272, only 1.9% of h13-24 gf readings above top edge 260. NE rate z=(r-.46)/.59, train max r=1.0 (bin 12); NE x2 -> r up to 2 (bin 16, z 2.6: out of training range).
- Insulin dose/rate z=(d-1.5)/1.0; bins: d>5 -> bin 16.
''')
print('saved')
```