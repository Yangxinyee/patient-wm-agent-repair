That output was mostly a printed default argument, so the tensor list I needed was cut off. Still, three things are now clear:
- The copy functions' source is lost.
- Their signatures show how to call them, for example `step_full(rws, E, htf, ...)` and `mha_m(...)`.
- The prelude FFN has a large hidden layer; the bias appears to have about 768 entries.

Before choosing where to inject the dose signal, I need the compact list of tensors, with shapes, for:
- the core;
- the coda;
- the heads that turn the state into glucose predictions.

Run 139 showed the recurrent-step copy is close to exact at drug bundles. The open design question is where a dose-surprise signal can enter the state so that it persists and is read by the glucose head. One candidate is E through b_in, the recurrent block and the coda; the other is a private channel read by the heads. The shapes of c_out, the coda and attr_heads will decide this. If the coda ends in a LayerNorm, or the heads are small linear maps, that pushes the choice one way or the other.

I am also writing the run-123-to-139 findings into the notebook now, so they are not lost again.

```python
notes_add = """
## Runs 123-143 (restart 4)
- Runs 123-139 built a CPU replica of a full core step at drug bundles (functions mha, ffn, age_emb, mha_m, build_mem, step_full,
  step_var, step_parts, coda_only, fitstat, step_cv; targets yt (2319,4,192) new slots, ot (2319,192) out from fwd_3 states).
  Variant comparison (run 139): memory cross-attn with age embedding added to keys AND values ('ageKV') fits best:
  err 39.0, R2 (y .998, out .994); lnM 210.8 (.879,.873); noage 252.5 (.836,.827); lnM_ageKV 78.6 (.998,.979).
- Source of replica functions not recoverable (no In history). Signatures: step_full(rws,E,htf,incl=True,unit=3600.,pre=False,dE=None).
"""
with open('/work/notes.md', 'a') as fh:
    fh.write(notes_add)
skip = ('embed.',)
rows = [f'{n}{tuple(SHAPES[n])}' for n in NAMES if not n.startswith(skip)]
print(len(rows), 'non-embed tensors')
print('; '.join(rows)[:6000])
print('\nembed:', '; '.join(f'{n}{tuple(SHAPES[n])}' for n in NAMES if n.startswith('embed.'))[:1500])
```