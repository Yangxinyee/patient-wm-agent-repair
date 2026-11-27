The architecture printout gave me what I needed. The variable listing was cut off because it was too long, so from now on I will keep outputs short.

What I now understand about the architecture:
- An insulin atom's rate and dose enter through bins and z-scores. They pass through `encode_action` into u_k, which reaches the state through the posterior z and the prelude, and then into the slots h.
- The value head reads `with_code`. That is a normalised view plus a "last value" register term, `last_numeric` applied to z_last_value and similar inputs.
- The rollout feeds the model's own drawn glucose back in. So any shift I inject into the prediction at hour 13 carries into later hours through the last-value register and the state. My edit's direct effect and the model's own persistence will combine, and the final gain will have to be calibrated with dose_edit passes on tuning stays.

From my notes:
- The original model ignores insulin dose magnitude.
- The true insulin effect is a transient impulse response: about −2.1 to −2.4 mg/dL per unit/h at lags 1–4, fading to 0 by lag 12. My preliminary readout truth on the sel100 rows is about −17.5 for x2 and +8.8 for x1/2.
- Norepinephrine raises MAP by about 8.8 mmHg in total per mcg/kg/min, spread over lags 1–5.

To avoid hurting factual NLL, the injected term should act on the dose relative to what the state already expects. The model already credits the glucose band with the usual drop.

Next I need the exact tensor names and shapes:
- `core.flow.a`, `core.a_in`, `core.b_in`, `core.delta_in`;
- `encode_action.*`;
- `attr_heads.value_bin`: linear or MLP?
- `last_numeric`, `query_norm`.

These decide where a dedicated memory channel can be placed. To keep the output small, I will print shapes only for the relevant groups and just counts for the rest.

```python
import collections
rel = ('core.flow','core.a_in','core.b_in','core.delta_in','core.c_out','core.out_norm','core.prelude_norm','encode_action','attr_heads','last_numeric','query_norm','embed.numeric','embed.bins','embed_transition','z_head','core.z_post','core.z_in','core.slots','core.coda')
out=[]; other=collections.Counter()
for nm_ in NAMES:
    if nm_.startswith(rel): out.append(f"{nm_}:{tuple(SHAPES[nm_])}")
    else: other[nm_.split('.')[0]+'.'+nm_.split('.')[1] if '.' in nm_ else nm_]+=1
print(' | '.join(out))
print(dict(other))
```