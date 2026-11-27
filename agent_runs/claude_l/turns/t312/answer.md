Run 284 established three facts:
- `W0z` is the original model with only `z_head` zeroed, which is the current edit.
- `mkW3(1, 1, W0z)` builds the extra-token edit, changing `z_in` and `z_post`.
- `evalW` returns, for each dose edit, (model shift, my record-based effect estimate, number of rows).

On 40 batch-1 stays at gain 1 the edit barely moves the answers:

| Dose edit | Model | Estimated effect |
|---|---|---|
| Insulin ×2 | −3.70 | −12.74 |
| Insulin ×½ | +1.68 | +6.37 |
| Norepinephrine ×2 | +1.12 | +5.77 |
| Norepinephrine ×½ | −0.96 | −2.89 |

Teacher-forced NLL improves slightly (1.53132 → 1.52995). So the edit is safe but about 3.5× (insulin) to 5× (norepinephrine) too weak.

The open question is how the readouts scale with gain. Run 281 found that the injected direction also changes attention scores. Large gains may therefore behave non-linearly: the ×2 effect could grow while the ×½ effect stalls, or the reverse.

Next I will:
- sweep the gains at 3, 6 and 10, recording all four readouts and the teacher-forced NLL;
- read which global variables `mkW3` uses (from its compiled code), so that I can swap in a better injected direction without its lost source.

If the readouts grow roughly in proportion to the gain, with ×2 and ×½ staying symmetric and NLL within budget, the existing builder can simply be scaled. Asymmetric or saturating growth would confirm the attention problem and mean the direction must be rebuilt.

```python
for f_ in [mkW3, keygrads, evalW]:
    c_=f_.__code__; print(f_.__name__, 'names:', c_.co_names[:60]); print('   vars:', c_.co_varnames[:40])
print('mkW3 consts:', [x for x in mkW3.__code__.co_consts if not hasattr(x,'co_code')][:40])
stays=list(BUN.keys())[:40]
SW={}
for G_ in [3.0,6.0,10.0]:
    Wg_=mkW3(G_,G_,W0z); r=evalW(Wg_,stays); n_=tfnll(Wg_,stays)
    SW[G_]=(r,n_); print('G',G_, {k:(v[0],v[1]) for k,v in r.items()}, 'tfnll %.5f'%n_[0])
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 7 (run 282+)\n- evalW(W,stays)->{1:insx2,2:insx.5,3:NEx2,4:NEx.5: (model,truth,n)}. base=W0z (orig+z_head0). 40 stays BUN[:40]: truth -12.74,+6.37,+5.77,-2.89. "
            "mkW3(1,1): -3.70,1.68,1.12,-.96, tfnll 1.52995 vs W0z 1.53132. Old Wt: -3.02,1.39,.73,-.89.\n- sweep: "+str({k:({kk:v[0][kk][0] for kk in v[0]}, round(v[1][0],5)) for k,v in SW.items()})+"\n")
```