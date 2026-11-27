The value-head replica is good enough for the edit. The remaining first-reading error is confined to the very first glucose_finger reading after admission (hour −1). That is irrelevant to dose effects at hours 13 to 24. On ordinary readings the replica tracks the tool's predicted means to about 1 unit.

The plan is to repurpose one or two hidden units of the value head. Each unit would be gated on the code embedding, so it fires only for glucose codes (or only MAP codes), and would add a dose-dependent tilt to the logits. That requires units that currently contribute almost nothing, so that repurposing them leaves the factual predictions unchanged.

I will compute each hidden unit's activation over all 5,214 readings, weighted by the norm of its output weights. If several units are effectively dead, the plan is feasible. If none is, I will need another injection point.

```python
e_full = ext_for()
q_ = Hln(view0_ + HT('type_emb.weight')[1], 'query_norm')
wc_ = q_ + HT('embed.code.weight')[codes_] + e_full
pre_ = Hlin(wc_, 'attr_heads.value_bin.0')
act_ = Fn.gelu(pre_)
w2n = HT('attr_heads.value_bin.2.weight').norm(dim=0)
contrib = (act_.abs() * w2n[None, :])
cmax = contrib.max(0).values; cmean = contrib.mean(0)
order_ = torch.argsort(cmax)
print('units with smallest max contribution (unit, max, mean, pre_max):')
for j in order_[:10].tolist():
    print(j, round(float(cmax[j]), 4), round(float(cmean[j]), 5), round(float(pre_[:, j].max()), 2))
print('median unit max contrib', round(float(cmax.median()), 3))
```