I found the real cause of the identical results: `PS` and `P_` are now the same object. `head_value` rebinds the global `P_` to whatever dict it is given, so `nllm(P_)` was really evaluating `PS`.

The evaluators are sound: `tf_eval` gives different NLL for the original (1.3990) and `PS` (1.4049). The scaffold's cost is therefore real: about +0.006 nats on sel2 (se about 0.003), already over the 0.005 budget. The individually measured dim costs were noise-selected (winner's curse), so they don't predict the combined cost.

Lowering the cost is now the main design problem. Zeroing a dim shifts LayerNorm statistics and removes its information. Dims with near-zero mean and small variance should be cheaper to repurpose, and mean-ablation (constant content) may cost less than zero-ablation.

Next I'll record this in the notebook, then compute per-dim mean and std of h from the tool's states (`HST`, sel2) and compare the current role dims with the cheapest-looking candidates.

```python
with open('/work/notes.md','a') as fh: fh.write('''
## Restart 6 (runs 154-175)
- Kernel intact but SOURCES LOST (no history). PS = pure KILL scaffold (zeros) of JALL dims [132,52,73,105,144,128,155,171]
  (ROLE jR132 jG52 jWI73 jSI105 jWN144 jSN128 jI155 jN171), value units VU=[12,107], prelude FFN units FFU=[342,601,696,737]; no wiring.
- BUG NOTE: head_value rebinds global P_ to its P argument (so P_ is just 'last dict used'). Evaluators are sound:
  tf_nll(P,ret=True)->(nll[5802],probs,mean,H,Out) on sel2 (needs global ages=age_h*3600); tf_eval(P, build_ds(tr,stays,0),True).
- Offline baseline tf_nll(PO)=1.40449 (tool 1.40394). SCAFFOLD PS COSTS +0.0060 (se .003) -> too expensive. Individual dimcosts were noise-selected.
- sel.json has sel1 (tuning, fwd1) and sel2 (train, fwd2/3). PO/P/Po/W/Wt = original dicts.
- From now on keep code in /work/*.py files.
''')
H4=HST.numpy() if torch.is_tensor(HST) else HST
mu=H4.mean((0,1)); sd=H4.std((0,1)); hb=H4.mean(1); sdb=hb.std(0)
print('h dims: |mean| median %.2f, sd median %.2f, min sd %.2f'%(np.median(abs(mu)),np.median(sd),sd.min()))
print('roles mean', np.round(mu[JALL],1).tolist(), 'sd', np.round(sd[JALL],1).tolist())
sc=np.sqrt(mu**2+sd**2); o=np.argsort(sc)
print('smallest rms dims', o[:12].tolist(), np.round(sc[o[:12]],1).tolist())
o2=np.argsort(sd); print('smallest sd dims', o2[:12].tolist(), 'sd',np.round(sd[o2[:12]],2).tolist(),'mean',np.round(mu[o2[:12]],1).tolist())
```