I now have the key facts:

- **Activation:** SiLU fits best for z_post.
- **Unit maps and code IDs:** recovered.
- **States from forward pass 2:** 4,932 bundles over 100 stays, 992 numbers each: `out` (192), `h` (4×192), `z` (32).
- **Bundle lists:** `BUN` holds them for those 100 stays.
- **Embedding:** includes an `embed.kq` term.

**Remaining unknowns in the core step:**
- how an absent u-token or e-token is handled in the prelude;
- what the cross-attention memory M holds;
- the units of the age input and how age enters the keys;
- the initial state;
- the gap input scaling.

**Plan.** Check one step at a time. Take the tool's state after bundle k−1, run my implementation of the step for bundle k, and compare with the tool's state after bundle k. Before writing that, I need the format of a `BUN` entry and the old gap lambda, so I can reuse my verified encoders.

```python
b0=BUN[2020000000]
print(type(b0),len(b0))
for item in b0[:4]: print(short(item,700))
print('...'); 
for item in b0[20:23]: print(short(item,700))
gl=z_pred_k.__defaults__
print(gl)
g_=gl[-1]; print(dis.code_info(g_.__code__)[:600]); dis.dis(g_)
print(inspect.signature(setenc), inspect.signature(empty_enc))
print(X[row_of[2020000000][0],:5], X[row_of[2020000000][0],192:197], X[row_of[2020000000][0],960:965])
```