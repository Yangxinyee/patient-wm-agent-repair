Run 204 printed the full architecture. Every path from a dose to the value predictions runs through the recurrent state:

action encoding u_k → z posterior and prelude tokens → E → slot update → coda → `out_norm(mean_slots(y))` → view → `query_norm` → value head.

This explains why my z-channel writer failed. Its signal entered as a small change against slot states of norm about 76. It was then renormalised by `out_norm` and `query_norm` before any head saw it.

A workable pathway needs two things. It must reach `y`/`out` along a direction large relative to its natural spread. It should also persist across steps with a controllable decay, from `core.flow.a` or `a_bar`.

Before designing it, I need the exact structure of `core.c_out`, `core.coda`, `core.b_in`, the prelude, `z_in` and `z_head`. Only then can I tell whether a mostly linear route exists from a prelude or z write to `y`. I'll list the core and head tensor names with shapes, compactly.

Some outcomes would confirm the plan: coda being a simple linear/LayerNorm/residual stack, or `b_in` and `c_out` being plain linear maps. A gated or MLP-heavy coda without a residual path would refute it. In that case I would switch to reading a dose direction already present in `out`.

```python
nm=[n for n in NAMES if n.startswith('core.') or n.startswith('z_') or 'query_norm' in n or 'out_norm' in n or 'type_emb' in n]
print(len(nm))
print('; '.join(f"{n}{tuple(SHAPES[n])}" for n in nm)[:3000])
```