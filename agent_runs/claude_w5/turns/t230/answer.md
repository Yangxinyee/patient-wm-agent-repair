The NLL problem is now clear, and it is not bad aim. The injected shifts reduce squared error everywhere (glucose MSE -0.91, MAP -0.25, lab -1.18), and the errors follow the shift with slope above 1. But the binned likelihood barely rewards moving the mean toward the observation: the first-order coefficient is about 0 for glucose against 0.0058 for a Gaussian. It still pays the quadratic cost.

The likely reason: the head predicts the next value largely from "last value" inputs that bypass the core state (last-value bin embedding, z-scored last value, a last-bin transition table). A belief change injected into the state can move the mean only by reshaping mass around that anchor. In the rollout this anchor follows the model's own drawn readings, so the shift should carry forward there.

Whether the current gains are near right depends on how the shift persists in the rollout, which only a forward pass can measure. Before spending one I do three things:
1. Check how the tool's per-reading NLL relates to my local one: 1.436 versus 1.048, possibly a constant log bin width per code.
2. Confirm that a float32 round-trip of the edited tensors leaves the injection unchanged.
3. Save the edit.

```python
# 1) local vs tool NLL per row (original model, same 100 stays)
f3k = f3.set_index(['stay_id', 'hour', 'event'])
tool_n, loc_n, cc = [], [], []
for j, i in enumerate(IDX):
    hh = H[i]; key = (int(hh['sid']), float(hh['hour']), hh['code'])
    try:
        v = f3k.loc[key, 'nll']
        if np.ndim(v) == 0: tool_n.append(float(v)); loc_n.append(n0[j]); cc.append(hh['code'])
    except KeyError: pass
tool_n, loc_n, cc = np.array(tool_n), np.array(loc_n), np.array(cc)
print('matched', len(tool_n), 'tool mean', tool_n.mean().round(4), 'local mean', loc_n.mean().round(4))
for c in np.unique(cc):
    m = cc == c; dlt = tool_n[m] - loc_n[m]; print('  ', c, 'tool-local mean', dlt.mean().round(4), 'sd', dlt.std().round(4), 'corr', np.corrcoef(tool_n[m], loc_n[m])[0,1].round(4))
# 2) float32 round trip check
WE32 = {k: (v.float().double() if k in ('core.prelude.0.ffn.0.weight','core.prelude.0.ffn.0.bias','core.prelude.0.ffn.2.weight','core.b_in.weight') else v) for k, v in WE.items()}
dA = run_cmp(WE, recs[:40]); dB = run_cmp(WE32, recs[:40])
print('float32 roundtrip: max rel diff of dh_new', max(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-9) for a, b in zip(dA, dB)).round(5))
# 3) save
for nm in ('core.prelude.0.ffn.0.weight','core.prelude.0.ffn.0.bias','core.prelude.0.ffn.2.weight','core.b_in.weight'):
    save_edit(nm, WE[nm].float())
print('edited', edited())
import pickle
pickle.dump(dict(wxI=wxI, bxI=bxI, wxN=wxN, bxN=bxN, wg=wg, bg=bg, q_I=q_I, q_N=q_N, vbar_I=vbar_I, vbar_N=vbar_N), open('/work/edit_parts.pkl', 'wb'))
```