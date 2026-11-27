The vocab gives the code ids:
- map_cuff = 5, glucose_finger = 6, map_arterial = 7;
- insulin = 8, norepinephrine = 9, glucose_lab = 10.

Normalisation is z = (x − median)/IQR, clipped to ±10, and the bins come from per-code edges. It does not say how the categorical tables are indexed (unit, rate_unit, route, action, status, kq, text).

A promising design is to make one dimension of h a clean leaky dose memory, in the style of a linear state-space model. That means zeroing that dimension's rows in the block and coda outputs, setting its decay through `core.flow.a` and `core.a_in`, and injecting through a row of `core.b_in`. The value head would then read it. The injection has to carry the dose relative to the band, and that can only be built and checked with a local copy of the encoder and prelude.

One check is available. The z-posterior means in the states let me verify a local u_k at action-only bundles. That needs the categorical indices for drug atoms. tensors.json may document them, so next I'll print its entries for the embedding tensors. I'll also record this session's findings in the notebook.

```python
open('/work/notes.md','a').write("""
## Session 2 (runs 38-46)
- Teacher-forced original (fwd_2, sel100): coef of next finger pred on insulin jitter dev = +0.38 (se .32) vs data -1.62 (se .66): model IGNORES dose magnitude.
- fwd_3 states at insulin bundles: band base R2 0.90, rate R2 0.65, jitter-dev R2 0.15 -> dose magnitude barely in state. Head-only edit can't work.
- Plan: dedicated h-dim leaky memory (zero its rows in blocks/coda outputs, set decay via flow.a/a_in, inject via b_in row), read by value head. Needs local reimpl of encoders+prelude; verify via z_post mean.
""")
TJ = json.load(open('/model/tensors.json'))
print(type(TJ), (list(TJ.keys())[:5] if isinstance(TJ,dict) else TJ[:2]))
ent = TJ.get('embed.categorical.unit.weight') if isinstance(TJ,dict) else None
print(json.dumps(ent)[:800] if ent else '')
```