This confirms the route is viable:
- **z is ignored everywhere.** At reading bundles the z token also gets zero attention in every head (heads 1 and 2 park on the empty-action token, head 3 reads e). So z_post and z_in can be repurposed at essentially no cost to the outputs.
- **Head 1 is the target.** At drug bundles head 1 attends 100% to e, which takes only 4 exact END-atom patterns (897, 783, 571 and 120 bundles). Its value varies strongly across patterns (spread 12 against a mean norm of 8), so the z token must reproduce the pattern-specific value.
- **Logit gaps.** To take over head 1 the z token's logit must rise by about 30 (gap −17 now), while head 3 needs care (z is currently only about 6 below the best token there).

The plan is a z token at drug bundles that copies e's head-1 value per pattern plus innovation × direction, wins head 1 and loses heads 0, 2 and 3. First I'll check the building blocks:
- (a) The existing z_post writer units (12 gate, 88 insulin, 105 NE): their pre-activations at drug versus non-drug bundles and their accuracy against the innovation targets.
- (b) The four exact e vectors and their within-pattern spread (should be about 0).

```python
W0o=A(load('z_post.0.weight')).astype(float); b0o=A(load('z_post.0.bias')).astype(float)
def zpre(ctx,e,u): return W0o@np.r_[ctx,e,u]+b0o
Pd=np.array([zpre(ctxs[i],*[np.asarray(x,float) for x in enc_e_var((int(sid[k]),round(float(hr[k]),3)),'none')])[[12,88,105]] for i,k in enumerate(kd)])
kr=np.where(typ==0)[0][::2]
Pr=np.array([zpre(np.asarray(ctx_from(xo[k-1],(hr[k]-hr[k-1])*3600),float).reshape(-1)[:192],np.asarray(e_read(k),float),np.asarray(U_EMPTY,float).reshape(-1)[:192])[[12,88,105]] for k in kr])
for j,nm in enumerate(['g12','a88','b105']): print(nm,'drug pre min/med/max %.1f %.1f %.1f | nondrug max %.1f'%(Pd[:,j].min(),np.median(Pd[:,j]),Pd[:,j].max(),Pr[:,j].max()))
print('a88 vs innovA (m=1): corr %.4f slope %.3f; b105 vs innovB corr %.4f slope %.3f'%(np.corrcoef(Pd[:,1]-Pd[:,0]*0,TA[2])[0,1],np.polyfit(TA[2],Pd[:,1],1)[0],np.corrcoef(Pd[:,2],TB[2])[0,1],np.polyfit(TB[2],Pd[:,2],1)[0]))
E4=np.array([np.asarray(enc_e_var((int(sid[k]),round(float(hr[k]),3)),'none')[0],float) for k in kd])
pid=np.unique(np.round(E4[:,:6],4),axis=0,return_inverse=True)[1].ravel(); print('n patterns',pid.max()+1,'counts',np.bincount(pid))
EP=np.array([E4[pid==p].mean(0) for p in range(pid.max()+1)]); print('within-pattern max dev', round(float(max(np.abs(E4[pid==p]-EP[p]).max() for p in range(len(EP)))),6))
```