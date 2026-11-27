Run 233 shows the injection fires 14 times per stay, yet the output states do not change at all. So either `step` ignores its z argument, or `run_cf` swaps in stored z values. Next I will call `step` directly on one insulin bundle with z and with z + Δ. A difference means `run_cf` overrides z. No difference means the replica's `step` does not use z.

```python
k=11; s=s0
o_=torch.tensor(xs[off[s]+k-1,:192],dtype=torch.float64); h_=torch.tensor(xs[off[s]+k-1,192:960],dtype=torch.float64).reshape(4,192)
e_,u_=bvd_orig(BUN[s][k][1]); e_=e_.double(); u_=u_.double(); dts=60.0*(BUN[s][k][0]-BUN[s][k-1][0])
Mem=torch.stack([bvd_orig(BUN[s][j][1])[0].double() for j in range(k+1)]); ages=torch.tensor([60.0*(BUN[s][k][0]-BUN[s][j][0]) for j in range(k+1)],dtype=torch.float64)
z0=zpost_orig(o_,h_,dts,e_,u_).double()
r1=step(o_,h_,dts,e_,u_,z0,Mem,ages); r2=step(o_,h_,dts,e_,u_,z0-0.7*gz_t.double()/gz_t.norm(),Mem,ages)
print(type(r1).__name__, [tuple(t.shape) for t in r1], 'diff', [float((a-b).abs().max()) for a,b in zip(r1,r2)])
```