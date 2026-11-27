The config shows 4 attention heads, 4 slots, a memory of 128, a 32-dim z, and an 8-column `last_numeric`. I'll keep printed output very short from now on, because long listings are being blocked. Next I'll record the current status in the notebook so it survives a restart.

```python
notes_txt = """
## Session 4 (runs 100-110)
- Old local step (no source recoverable) gives step_err3('empty_const') = (5.255, 0.0614) -> ~6% rel error, NOT validated.
- Config: d192, heads 4, ff x4, prelude1, recurrent2, coda1, slots4, mem128, z32. coda/prelude = pre-norm layers (norm1, attn, norm3, ffn).
- Design: dedicated dim j: injection only via prelude attention (u_k token) -> E -> b_in row j; persistence: zero writes to j in blocks/coda/c_out, gamma_j=0 in norms; readout via out_norm/query_norm -> value_bin.0 column j.
- Keep outputs SHORT (content filter).
"""
open('/work/notes.md','a').write(notes_txt); print('ok')
```