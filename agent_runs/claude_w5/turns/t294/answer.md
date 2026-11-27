`run_tf` returns one 6-tuple per bundle, presumably (out, y, z, E, hf, h_new), so the state after a bundle is out = el[0], h = y = el[1] and z = el[2].

This run compares local and tool states on all 13 stays, for both models, split by bundle type. It also compares the size and direction of the edit's effect, local (edited − original) against tool.

```python
import numpy as np, time
def loc_states(W, B):
    r = run_tf(B, step=step_with(W))
    return np.stack([np.concatenate([np.asarray(t[0],float).ravel(), np.asarray(t[1],float).ravel(), np.asarray(t[2],float).ravel()]) for t in r])
t0 = time.time(); rowsC = []
for sid in ids5:
    B_t = make_bundles(tun[tun.stay_id==sid]); m_ = sid7==sid
    Le, Lo = loc_states(WE, B_t), loc_states(W64, B_t)
    Te, To = X7e[m_], X7o[m_]
    hi, hn = z7o['has_insulin'][m_].astype(bool), z7o['has_norepinephrine'][m_].astype(bool)
    for k in range(len(Te)):
        rowsC.append({'sid': sid, 'k': k, 'drug': bool(hi[k] or hn[k]),
                      'err_e': np.linalg.norm(Le[k]-Te[k])/np.linalg.norm(Te[k]), 'err_o': np.linalg.norm(Lo[k]-To[k])/np.linalg.norm(To[k]),
                      'nd_tool': np.linalg.norm(Te[k]-To[k]), 'nd_loc': np.linalg.norm(Le[k]-Lo[k]),
                      'cos': float((Te[k]-To[k])@(Le[k]-Lo[k])/(np.linalg.norm(Te[k]-To[k])*np.linalg.norm(Le[k]-Lo[k])+1e-12)),
                      'err_e_z': np.linalg.norm(Le[k,960:]-Te[k,960:])/np.linalg.norm(Te[k,960:]), 'err_o_z': np.linalg.norm(Lo[k,960:]-To[k,960:])/np.linalg.norm(To[k,960:])})
DC = pd.DataFrame(rowsC); print('time %.0f' % (time.time()-t0))
print(DC.groupby('drug')[['err_o','err_e','err_o_z','err_e_z','nd_tool','nd_loc','cos']].median())
print('worst err_e rows:'); print(DC.sort_values('err_e').tail(5)[['sid','k','drug','err_e','err_o','nd_tool','nd_loc','cos']])
print('err_e by k (first 12 bundles, median):', DC.groupby('k').err_e.median().head(12).round(6).to_dict())
```