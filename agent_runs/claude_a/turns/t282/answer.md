The cause is in my evaluation, not the build. `build_ins2` writes the rate into the dose encoder's output, but `tf_nll` and `tf_eval` reuse dose encodings precomputed with the original weights. So the edited encoder never ran: the memory received the original dims' content (which happens to correlate −0.69 with the rate), and the identical costs simply reflected unchanged inputs. The real model re-encodes doses with the edited weights. Offline, I must therefore re-encode each dataset with `reenc(ds, PA)` before scoring. This run re-encodes sel2 and dsE, re-measures the cost, and redoes the memory-fidelity check on the re-encoded states. I expect the memory to track the ideal decayed dose sum closely (correlation ≥0.95). The cost is a fresh unknown now that the staging actually runs.

```python
dsA2=reenc(ds_s2,PA); dsAE=reenc(dsE,PA)
rr=tf_eval(PA,dsA2,True); print('tf_eval ret types', [type(x).__name__+str(tuple(getattr(x,'shape',()))) for x in (rr if isinstance(rr,(tuple,list)) else [rr])][:6])
nA=rr[0] if isinstance(rr,(tuple,list)) else rr
print('cost sel2 %+.5f  dsE %+.5f'%(float((nA-a_s2).mean()), dcost(PA,dsAE,a_E)))
Hx=[x for x in rr if hasattr(x,'shape') and len(x.shape)==3 and x.shape[-1]==192]
if Hx:
    memA=Hx[0][...,50].mean(1).numpy() if hasattr(Hx[0],'numpy') else Hx[0][...,50].mean(1)
    cfA=np.linalg.lstsq(A_,memA,rcond=None)[0]; rsA=memA-A_@cfA
    print('mem~m*: slope %.3f icpt %.3f corr %.4f resid sd %.3f | mem sd %.2f max %.1f'%(cfA[0],cfA[1],np.corrcoef(mstar,memA)[0,1],rsA.std(),memA.std(),np.abs(memA).max()))
```