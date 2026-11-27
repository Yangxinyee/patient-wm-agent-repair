fwd_7 shows the saved v4 edit fails on the tool, so it cannot stay.

| | Original | v4 edit |
|---|---|---|
| Value NLL per reading | 1.519 | 46.3 |
| glucose_finger mean abs error | 10.8 | 99 |
| map_arterial mean abs error | 4.0 | 38 |

Lab and cuff readings also broke, although my replica said those rows were untouched. So the tool does not compute my edited units the way my replica does.

My hypothesis is numerical precision. The gate used the 1/gamma (LayerNorm) cancellation trick, which needs exact arithmetic. If the tool's GPU matmuls run in reduced precision (bf16, fp16 or TF32), large-magnitude gate weights produce garbage pre-activations.

Next run:
- Keep a copy of the v4 tensors in memory and drop all edits, so the current submission is the safe original model.
- Measure the edited units' weight magnitudes against the originals.
- Recompute those units' pre-activations in bf16 and fp16 and compare with fp32.

Large errors (several units or more) in reduced precision would confirm the hypothesis. Then the next design must keep weights at the original scale and avoid exact-cancellation tricks. Small errors would refute it and point to a structural mismatch between replica and tool. I'll also print what helpers and variables the kernel still holds.

```python
import inspect, torch
print('type', type(load('attr_heads.value_bin.0.bias')))
E_v4 = {n: np.array(load(n)) for n in edited()}
for n in list(edited()): drop_edit(n)
print('edited now', edited())
qn=[n for n in NAMES if 'query_norm' in n]; print(qn)
for n in qn:
    t=np.asarray(load_original(n)); print(n, t.shape, 'absmin %.4g absmax %.4g'%(np.abs(t).min(), np.abs(t).max()))
W1o=np.asarray(load_original('attr_heads.value_bin.0.weight')); b1o=np.asarray(load_original('attr_heads.value_bin.0.bias'))
W1e=E_v4['attr_heads.value_bin.0.weight']; b1e=E_v4['attr_heads.value_bin.0.bias']
U=[114,10,0,107]
print('orig W1 row norms median %.3g max %.3g; |b1| max %.3g'%(np.median(np.linalg.norm(W1o,axis=1)),np.linalg.norm(W1o,axis=1).max(),np.abs(b1o).max()))
for u in U: print(u,'edit |w| %.3g max %.3g b %.4g'%(np.linalg.norm(W1e[u]),np.abs(W1e[u]).max(),b1e[u]))
X=torch.as_tensor(np.asarray(Xf),dtype=torch.float32); Wt=torch.as_tensor(W1e[U],dtype=torch.float32)
p32=(X@Wt.T).numpy()+b1e[U]
for dt in [torch.bfloat16, torch.float16]:
    pl=(X.to(dt)@Wt.to(dt).T).float().numpy()+b1e[U]; err=np.abs(pl-p32)
    print(dt,'err med',np.round(np.median(err,0),3),'p99',np.round(np.percentile(err,99,0),2))
cfa=np.asarray(cf)
for c in range(4):
    m=cfa==c; print('code',c,'pre32 mean',np.round(p32[m].mean(0),1),'sd',np.round(p32[m].std(0),2))
g=[k for k,v in globals().items() if not k.startswith('_') and not callable(v) and not inspect.ismodule(v)]
print(len(g), ' '.join(sorted(g))[:1500])
print('build_v4', inspect.signature(build_v4) if 'build_v4' in globals() else None)
```