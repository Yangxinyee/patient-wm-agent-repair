The z_in columns are not weak: every one, z18 and z28 included, has norm about 1.6. That compares with a typical z-token norm of 2.5. So the fwd8 writes (about 1–2 units) changed the z token heavily, yet the next ctx moved by only 0.04.

There are two possible explanations:
1. The prelude barely attends to the z token, so z is weak as an input to the state.
2. The z write does reach the h slots, but in directions that cancel in the slot mean. Both ctx and the heads' view read only `mean_slots(h)` and `out`, so they would not see it, while the prelude and blocks of later steps would.

The fwd8 states can tell these apart. Take the state right after each drug bundle and compare edited with original, split into out, the four h slots, the slot mean and z. Compare each change with the natural spread of that block, and check how linear it is in the rates written at that bundle.

- If Δh per slot is large but Δmean_slots and Δout are tiny, the signal exists and only the readout is blind. That points to a slot-level design.
- If everything is tiny, z is effectively a dead input, and the dose signal must enter through the action token u or the prelude itself.

```python
kd8=k8  # drug-bundle rows (state after the drug bundle)
dX=xe[kd8]-xo[kd8]; Xo_=xo[kd8]
blocks_={'out':slice(0,192),'h0':slice(192,384),'h1':slice(384,576),'h2':slice(576,768),'h3':slice(768,960),'z':slice(960,992)}
for nm,sl_ in blocks_.items():
    print('%-4s |dX| %.4f   natural sd-norm %.3f'%(nm,np.linalg.norm(dX[:,sl_],axis=1).mean(), np.sqrt(Xo_[:,sl_].var(0).sum())))
hs_o=Xo_[:,192:960].reshape(-1,4,192); hs_d=dX[:,192:960].reshape(-1,4,192)
print('mean_slots(h): |d| %.4f natural %.3f'%(np.linalg.norm(hs_d.mean(1),axis=1).mean(), np.sqrt(hs_o.mean(1).var(0).sum())))
# linearity in written rates at this bundle (dz18=.29*Dins, dz28=1.83*Dne) plus const
Xl2=np.c_[np.ones(len(kd8)),Di,Dn]
for nm,sl_ in [('out',slice(0,192)),('h',slice(192,960)),('z',slice(960,992))]:
    B2,*_=np.linalg.lstsq(Xl2,dX[:,sl_],rcond=None); rs=dX[:,sl_]-Xl2@B2
    print(nm,'R2 lin in rates %.3f'%(1-rs.var(0).sum()/dX[:,sl_].var(0).sum()),' per-unit norms: ins %.4f ne %.4f const %.4f'%tuple(np.linalg.norm(B2[[1,2,0]],axis=1)))
print('dz18 at drug rows vs .29*Dins: corr %.3f'%np.corrcoef(dX[:,960+18],Di)[0,1], ' dz28 vs Dne corr %.3f'%np.corrcoef(dX[:,960+28],Dn)[0,1])
# reading bundle immediately before (state after reading bundle): response to constant writes
dXr=xe[kd8-1]-xo[kd8-1]
print('reading rows |dout| %.4f |dh| %.4f |dz| %.4f'%(np.linalg.norm(dXr[:,:192],axis=1).mean(),np.linalg.norm(dXr[:,192:960],axis=1).mean(),np.linalg.norm(dXr[:,960:],axis=1).mean()))
```