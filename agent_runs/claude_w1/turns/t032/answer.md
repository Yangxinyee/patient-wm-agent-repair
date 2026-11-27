The listing shows the paths dose information can take to the glucose and MAP predictions:
- **Atom embedding:** the dose enters through `embed.numeric`, whose z_dose and z_rate columns are shared by both drugs, and through `embed.bins.rate`/`dose`.
- **Action encoder:** an MLP set encoder ending in a LayerNorm.
- **Core:** z_post, the prelude transformer, the recurrent update and the coda.
- **Heads:** the value head is a 2-layer MLP (`attr_heads.value_bin`, 192 → 192 → 17 logits) that reads q plus a register of the previous reading's value.

Before deciding where to edit, I want to know whether the model's existing (tiny) dose response has the right time profile and is just too small. If so, scaling it up might roughly work. If its shape is wrong, I need to build a pathway with the right shape.

Next I'll compare, hour by hour (13–24) and pooled over reached rows, the model's shift with my estimated true shift for each edit. I'll also see how the per-stay model effect scales with the stay's dose.

```python
for edit in ['insulin_x2','insulin_half','ne_x2','ne_half']:
    m = res[edit]
    byh = m.groupby('hour').agg(model=('shift','mean'), true=('true','mean'), n=('shift','size'))
    byh['ratio'] = byh.model/byh.true
    print(edit); print(byh.round(3).T.to_string())
# per-stay: model vs true
m = res['insulin_x2']
ps = m.groupby('stay_id').agg(model=('shift','mean'), true=('true','mean'), n=('shift','size'))
print(np.corrcoef(ps.model, ps.true)[0,1], np.polyfit(ps.true, ps.model, 1))
m = res['ne_x2']
ps2 = m.groupby('stay_id').agg(model=('shift','mean'), true=('true','mean'), n=('shift','size'))
print(np.corrcoef(ps2.model, ps2.true)[0,1], np.polyfit(ps2.true, ps2.model, 1))
```