The numbers support the dedicated-channel plan:
- The normalised slot tokens are confined to roughly 40 dimensions: 124 eigenvalues are below 1e-4 and 152 below 1e-3. That leaves plenty of nearly unused directions for a clean channel.
- `c_out` and `b_in` have singular values from about 5 down to near 0. Their top ~100 directions are usable.
- The news directions are the same vector in all four slots (cos 1.0). Per reading unit: finger 0.82/mg/dL, lab 0.92, arterial 5.6/mmHg, cuff 3.0.
- One caveat: the slot-token σ ranges from 27 to 57, so a channel through `prelude_norm` will have a state-dependent gain of about 1.5× between the 10th and 90th percentile.

Next I need to rebuild the input side. That means the action encoder (u_k), the END-atom encoder for e_k at drug bundles, and the prelude attention and FFN. With these I can identify the insulin, NE and presence readouts in the v3 key/value edit, and measure the natural variance of candidate channel directions after the attention and FFN.

This run collects what the encoders need: embedding/encoder tensor names, vocab codes, kinds and normalisation for insulin and NE. It also records the last findings in the notebook. If the tables match my notebook formula (code + kind + ekind + unit + rate_unit + kq + bins + numeric), I can rebuild the exact encoder next run.

```python
note = """
- Run 255: v3 effect on state is tiny: median |d out| .004 (out norm 13.9), |d mean_h| .2-.3 at drug bundles (h slot norm 560-1100).
- Run 256: softplus(delta_in)~1 (.90-1.07), a_bar .30-.46 (h keeps ~37%/step); core.slots norm <1 (slot tokens ~= decayed h).
  b_in, c_out sv 5.5..0.001 (top ~100 dirs usable). prelude_norm w .8-1.1, b ~0. Value head reads out=LN(mean y) and LN(mean h flowed).
- Run 257 (an1.npz): LN slot-token sigma 27-57 (median 40) -> channel via prelude_norm has 1/sigma gain variation.
  cov(LN slot tokens): 124 eig<1e-4, 152<1e-3 (state lives in ~40 dims) -> many clean channel directions.
  News directions (dY per unit surprise, same vector in all 4 slots, cos 1.0): finger .82/mgdl, lab .92, art 5.6/mmHg, cuff 3.0;
  cos finger-lab .51, art-cuff .905, finger-art -.18. Saved nu (event,slot,dim) order [finger,lab,art,cuff].
- PLAN v4 'dedicated channel': head-1 out-proj maps insulin readout -> c*q (small c, q in slot null space, zero-mean, low FFN sensitivity);
  core.b_in += t (x) w reads q from E and writes t = C_out^-1(-k*nuG) (b_in has no bias; w.E_nat must be ~0). Same for NE with nuM(art).
"""
open('/work/notes.md','a').write(note)
for n in NAMES:
    if n.startswith(('embed','encode.','z_in','z_post','z_prior','core.age')) : print(n, SHAPES[n])
print('codes', VOCAB['codes']); print('atom_kinds', VOCAB['atom_kinds']); print('event_kinds', VOCAB['event_kinds'])
nm = VOCAB['numeric']; print(type(nm), list(nm.keys())[:20] if isinstance(nm,dict) else nm[:3])
for k in ['insulin','norepinephrine']:
    if isinstance(nm,dict) and k in nm: print(k, nm[k])
print(VOCAB['numeric_note'])
```