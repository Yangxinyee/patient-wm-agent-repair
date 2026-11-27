The kernel still has my per-stay-hour comparison table `R`: model shift vs true shift for each edit. The true shift comes from parametric impulse responses fitted earlier, which match the estimated ones closely. `KI` is -2·k·0.6^(k-1) per unit/h of insulin. `KN` is 3.3·k·0.39^(k-1) per mcg/kg/min of norepinephrine.

The first rows show the gap. For one stay, the model predicts -0.31, -0.37, -0.67 under insulin x2, while the truth is -2.0, -4.4, -5.6.

Next I'll summarise `R` by edit (mean model vs mean true, and their ratio). I'll also write the architecture facts and the forward-pass-1 results into my notebook so a restart can't lose them. I expect the true insulin effect to be several times the model's and the true norepinephrine effect to be a few times the model's. That would confirm that the edit must scale up both drugs' effects, insulin by more.

```python
s1 = R.groupby('edit')[['model','true']].agg(['mean','count'])
print(s1.to_string())
for e in R.edit.unique():
    r = R[R.edit==e]
    b = np.polyfit(r.true, r.model, 1)
    print(e, 'slope model~true', b.round(3), 'corr', np.corrcoef(r.true, r.model)[0,1].round(3))
add = """
## Architecture summary (from /model/ARCHITECTURE.md)
- atoms -> embed (code+kind+unit+bins.value/dose/rate (17 rows shared across codes, per-code edges) + numeric@[zv,hv,zd,hd,zr,hr]+bias)
- measurements -> encode.* (DeepSets) -> e_k ; insulin/NE atoms -> encode_action.* -> u_k (0 if none)
- state: h (4 slots x192), memory M (128 last e's), out(192). Step: time-flow decay h*=exp(-dt*exp(core.flow.a));
  z posterior from [ctx,e_k,u_k]; prelude transformer over [h+slots, e_k, u_k, z_in(z)] -> E (prelude_norm);
  blocks(h+E) -> u ; h <- a_bar*h + softplus(delta_in)*b_in(E) + (u-(h+E)); coda: y=coda(c_out(h)); h=y; out=out_norm(mean_slots(y))
- heads: view = out + out_norm(mean_slots(h after flow)) + gap(log dt) + z_head(z); q=query_norm(view+type_emb[type]);
  with_code = q + embed.code[code] + last_numeric([has_last, log(1+age)/10, z_last_value, z_last_rate]) + bins.value[last]+bins.rate[last]+last_status
  value logits = attr_heads.value_bin(with_code) + embed_transition.value_bin[code*17+last bin]; mean = sum p*centre.
- dose_edit: record to hour 12 as is; from 13 readings hidden & sampled; insulin atoms from hour 12 on get rate&dose x2 (z and bins change).
## Forward pass 1 (original, dose_edit, tuning, 100 stays in sel1.json)
- model: ins x2 -2.75, ins x1/2 +1.38, NE x2 +0.69, NE x1/2 -0.58 (n stay-hours 818 / 693)
- truth table R (kernel) uses KI=-2k*0.6^(k-1), KN=3.3k*0.39^(k-1) (per unit/h resp. per mcg/kg/min, per infusion hour)
"""
open('/work/notes.md','a').write(add)
```