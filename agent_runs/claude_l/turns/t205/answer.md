Two useful facts came out of this:

- **z mean only:** the state update uses the z posterior mean, so a z-channel edit is deterministic. The log-sd outputs don't matter for the rollout.
- **Replica hook:** `bstep_z` takes a `zfn(ctx, e, u, info)` that returns the z token directly, with `info` holding the true insulin and NE rates of the bundle.

`res_tf` holds three teacher-forced experiments, which I ran but whose results were never delivered:
- `dup`: the z token is an exact copy of u;
- `ki4` and `ki8`: presumably the copy plus an injected insulin residual at gain 4 and 8.

The other recent variables point to the same plan:
- the policy decoders `w_ins`, `w_ne`, `pi_i`, `pi_n`;
- the u grid `Ug`, used to define an injection direction;
- a propagation Jacobian `Js` over 42 steps, with decode reliability 0.887 for one direction.

Next I'll summarise `res_tf` compactly:
- per variant, the mean value NLL compared with the baseline (from the stored factual NLL of the same readings);
- how the predicted finger glucose shifts with the insulin residual at lags 1–3.

Expected outcomes:
- `dup` should match the baseline almost exactly (identical tokens only split attention). If so, the copy-u trick is safe.
- `ki4`/`ki8` should show a larger lag-1 response with little NLL change. If NLL worsens a lot, the injection direction is wrong.

```python
for k,v in res_tf.items():
    print(k, type(v).__name__, len(v), type(v[0]).__name__ if len(v) else '', (len(v[0]) if hasattr(v[0],'__len__') else v[0]) if len(v) else '')
    try:
        if isinstance(v,dict): print(list(v.keys())[:5])
        else:
            x=v[0]; print(' first elem:', (x[:2] if isinstance(x,(list,tuple)) and len(x)>2 else x))
    except Exception as ex: print('err',ex)
print(type(pi_i[0]).__name__, getattr(pi_i[0],'shape',None), type(pi_i[1]).__name__, getattr(pi_i[1],'shape',None))
print('w_ins', tuple(w_ins.shape), 'w_ne', tuple(w_ne.shape), 'cf', np.round(np.linalg.norm(cf,axis=1),3))
print('cur[0]', cur[0] if len(cur) else None, ' idx[0]', idx[0], ' fit[:3]', np.round(fit[:3],3))
```