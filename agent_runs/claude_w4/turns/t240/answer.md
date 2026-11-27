Now I know exactly how the prelude computes. It is one self-attention block over 7 tokens (4 slots, event e, action u, z), followed by an FFN. Its output E is the prelude-normalised slot positions.

The cached CPU inputs for all 2,319 drug bundles are still in memory:
- the raw token stacks `TOK`, the teacher-forced previous slots `htf`, and the prelude outputs `E0`;
- the target new slots `yt` and target `out` vectors `ot` (from the forward states);
- similar arrays for 520 non-drug bundles (`TOKn`, `En`, `attn_n`).

So I can compute exactly how a vector added to the head-1 output changes `out`, and therefore the next predicted glucose or MAP.

This also points to a cleaner v4 carrier with no output remapping. I would add a rank-1 term a·(r·x) to the head-1 values, where the readout r is orthogonal to every non-action token, so nothing leaks. I would leave the head-1 output weights unchanged. The slots then receive W1o·a·rate, and a (48-dim) can be chosen through the Jacobian to land on the glucose or MAP readout.

First I must call the replica correctly. Next I'll:
1. Read how `build_mem` uses its `rws` argument.
2. Look at candidate row-index arrays.
3. Run `step_full` on a few drug bundles to see whether it reproduces `yt`/`ot`, and how long it takes.

If the error is small (it was R² 0.998 before), I'll compute the Jacobian next.

```python
c=build_mem.__code__; print('build_mem', c.co_varnames[:c.co_argcount], build_mem.__defaults__, '\n names', c.co_names)
t=brief(build_mem); print(' ; '.join(x.split(' ',1)[-1] if x.startswith(('LOAD_FAST','LOAD_GLOBAL','LOAD_ATTR','LOAD_CONST','STORE_FAST')) else x.split(' ')[0] for x in t)[:1800])
for nm in ['drug_rows4','d_idx','pos','di','mi','dr3','bd']:
    v=globals()[nm]; print(nm, getattr(v,'dtype',None), np.asarray(v)[:6])
```