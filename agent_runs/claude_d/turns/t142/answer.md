That's the breakthrough. The best layout puts `has_last` in column 0, `log(1+age)/10` in column 3 and the last z-score in column 6, keeping the last-bin embedding. With it, the reproduction matches the factual CSV to 2.03 mg/dL RMS for glucose and 1.07 for MAP. That equals the sampling noise of the model's 4 z draws, so my offline value head is now effectively exact.

This changes the plan. I can fit decoders on states, design the value-head edit offline, and compute its factual NLL cost myself, saving forward passes for calibrating the dose_edit response. Next I'll record this in the notebook so it survives a restart, and check the NE record timing for the MAP side.

```python
notes_add = """
## Restart 4 (runs 108-129): EXACT HEAD REPRODUCTION FOUND
- Bundles = WORLD atoms at hour + END atoms at end_hour (insulin/NE records AND admission rows), grouped by minute -> matches states rows exactly.
- view = out + out_norm(mean_slots(h*exp(-dt*exp(flow.a)))) + gap(log(dt_s)/10) + z_head(z), z~prior(ctx), ctx=view w/o z_head. GELU everywhere, LN eps 1e-5.
- q = query_norm(view + type_emb[1]) (type = akind*5+ekind; measurement WORLD = 1).
- with_code = q + embed.code[code] + last_numeric(f8) + last_numeric.bias + embed.bins.value[last bin] + embed.bins.rate[0] + last_status[0]
  f8: col0 = has_last, col3 = log1p(age_seconds)/10, col6 = z_last_value (same code, strictly earlier minute); other cols 0 for measurements.
- logits = value_bin.2(gelu(value_bin.0(with_code))) + embed_transition.value_bin[code*17+last bin]; bins: 1+searchsorted(edges, x, 'left').
- Repro vs fwd4 CSV: rms 2.03 (gluc) / 1.07 (map) = 4-draw z sampling noise floor (1.96/1.08). => offline head is exact.
- Plan: rank-1/2 update of attr_heads.value_bin.0 (W1 += eps_vec w^T) with decoder w of dose surprise from with_code, eps_vec chosen so mean shifts only for glucose (ins) / MAP (NE). Calibrate scale with dose_edit 'both' on tuning sel1.
"""
open('/work/notes.md','a').write(notes_add)
ne = TRAIN[TRAIN.event=='norepinephrine']
print('NE minute-of-hour', (np.round(ne.hour*60)%60).value_counts().head(5).to_dict())
ins = TRAIN[TRAIN.event=='insulin']; print('INS minute-of-hour', (np.round(ins.hour*60)%60).value_counts().head(3).to_dict())
print('readings minute-of-hour', (np.round(TRAIN[TRAIN.event.isin(list(NUMV))].hour*60)%60).value_counts().head(3).to_dict())
```