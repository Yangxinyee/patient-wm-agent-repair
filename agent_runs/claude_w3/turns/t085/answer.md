The config fixes the structure: 4 heads, one pre-norm prelude layer, two recurrent layers with self- and cross-attention, one coda layer, and LayerNorms everywhere.

An exact reimplementation would need many unknown details of the atom embedding, which is risky. A cheaper route is "dimension surgery": repurpose one or two little-used residual dimensions as a dedicated dose memory.
- Carrier and reading: one dimension carries a dose signal from the action token through a prelude attention value channel into h. The blocks and coda leave it untouched, its decay is set by `flow.a`, `a_in` and `delta_in`, and the value head reads it.
- Calibration: the overall gain is calibrated with dose_edit passes.

The cost is losing whatever those dimensions carried, so I first need dimensions the model barely uses. Next I will score each of the 192 residual dimensions by the LayerNorm gains and the projection weights that touch it. If a few dimensions have clearly smaller usage than the rest, surgery is plausible. If usage is uniform, the factual-fit cost may be too high and I need another plan.

```python
def nrm(n_): return load_original(n_).float()
ln_names = [n_ for n_ in NAMES if (n_.startswith('core.') and ('norm' in n_) and n_.endswith('weight'))]
G = torch.stack([nrm(n_).abs() for n_ in ln_names])          # (L,192)
outp = [n_ for n_ in NAMES if n_.startswith('core.') and (n_.endswith('attn.out.weight') or n_.endswith('cross.out.weight') or n_.endswith('ffn.2.weight'))]
O = torch.stack([nrm(n_).norm(dim=1) for n_ in outp])        # rows written
use = G.mean(0)*O.mean(0)
print(len(ln_names), len(outp))
print('gamma mean range', float(G.mean(0).min()), float(G.mean(0).median()), float(G.mean(0).max()))
o_ = torch.argsort(use)[:8]
print('lowest use dims', o_.tolist(), np.round(use[o_].numpy(),3), 'median', round(float(use.median()),3))
print('flow.a tau_h of those', np.round((1/torch.exp(nrm('core.flow.a'))/3600)[o_].numpy(),2))
```