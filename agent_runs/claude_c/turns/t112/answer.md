The reading-bundle posterior z does encode reading surprises:
- Glucose_finger surprise loads mainly on z dims 13, 10, 21 and 30; a 1-sd surprise moves z by about 0.33.
- MAP_arterial surprise loads on dims 12 and 13.
- The glucose and MAP directions are distinct (cosine −0.24). The lab and cuff readings share them (cosine 0.84 and 0.92).

So a glucose-surprise direction g and a MAP-surprise direction m exist in z-space.

Next is the potency test. I clear my current edits and amplify the z token's response along g and along m by four times (rank-1 on the `z_in` input side). Then a factual pass with both models on the same 100 tuning stays shows whether next-hour predictions move.

If the z token feeds the state update, the edited model will over-react to surprises: the change in prediction at t+1 will regress clearly on the surprise at t, by several mg/dL. A near-zero slope would confirm that the z token is ignored, and I would abandon the z route entirely. I also record the new findings in the notebook.

```python
clear_ops()
np.save('/work/dir_g.npy', gU); np.save('/work/dir_m.npy', mU)
rank1('z_in.weight', gU, 3.0, side='in')
rank1('z_in.weight', mU, 3.0, side='in')
print(ops())
print(len(stays100), stays100[0], stays100[-1])
open('/work/notes.md','a').write("""
## Restart-5 (runs 91-97)
- Slots of h are identical copies (rel diff 1e-4); h norm ~842 (accumulator), out=LN(h) norm ~13.9.
- Pass-5 (numeric x5) diffs: per-step injection along dose dirs (ins sd 4.5, ne 21.6) << natural step sd along same dirs at reading bundles (~100-110) -> NO selective direction in h; state-space edits (b_in/c_out) ruled out.
- u_k is LN'd (encode_action.norm) and re-LN'd (prelude norm1) -> input scaling only rotates token -> saturates.
- z at reading bundles encodes surprises: glucose_finger dir g (dims 13,10,21,30; 0.33 per sd), map_arterial m (dims 12,13); saved /work/dir_g.npy, dir_m.npy.
- z route blocker: z26 mixes insulin & NE (cross-effects would cost huge factual NLL). Testing z-token potency first (pass 10: z_in x4 along g and m, factual both).
""")
```