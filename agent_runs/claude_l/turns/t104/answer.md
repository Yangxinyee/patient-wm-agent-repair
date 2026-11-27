Even with 20,000 Levenberg–Marquardt evaluations the fit does not converge (status 0, RMS 0.008). Indirect fitting of the hidden context cannot tell me which component is wrong, so I am stopping it.

A much more direct method is available: a **probe edit**. I temporarily set `z_post` to a pass-through. 32 hidden units get a large positive bias (+30), which puts any of the candidate activations in its linear regime, and each unit computes a known random projection of the z_post input [ctx, e, u]. The output layer removes the bias. The "z posterior mean" the tool returns is then exactly 32 projections of the model's true ctx, e and u at every bundle: 10 for ctx, 11 for e and 11 for u. The edited model's own out/h are returned too, so I can compute ctx from them.

One states pass therefore tells me whether my encoders and my ctx formula are right, and which of them is wrong. This pass only diagnoses; I will drop the edit afterwards. Next I build and save this edit, and print the stay list for the request.

```python
exec(open('/work/setup.py').read()) if 'W0' not in globals() else None
rs=np.random.RandomState(123)
P=np.zeros((32,576))
for j in range(32):
    blk=0 if j<10 else (1 if j<21 else 2)
    v=rs.randn(192); v/=np.linalg.norm(v); P[j,192*blk:192*(blk+1)]=v
np.save('/work/probe_P.npy',P)
W1n=np.zeros((192,576),np.float32); W1n[:32]=P; b1n=np.zeros(192,np.float32); b1n[:32]=30.0
W2n=np.zeros((64,192),np.float32); W2n[:32,:32]=np.eye(32); b2n=np.zeros(64,np.float32); b2n[:32]=-30.0
save_edit('z_post.0.weight',W1n); save_edit('z_post.0.bias',b1n); save_edit('z_post.2.weight',W2n); save_edit('z_post.2.bias',b2n)
print(edited())
bj=json.load(open('/work/batches.json')); print(type(bj), list(bj.keys())[:5] if isinstance(bj,dict) else len(bj))
b1_=bj['batch1'] if isinstance(bj,dict) and 'batch1' in bj else None
print('stays==batch1?', b1_ is not None and sorted(b1_)==sorted(stays))
print(json.dumps([int(s) for s in stays]))
```