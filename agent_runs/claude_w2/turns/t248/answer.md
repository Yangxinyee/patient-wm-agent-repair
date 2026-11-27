The store came back empty. In teacher-forced mode the engine only evaluates the head where values are hidden or `want_nll=True` is set, and I had not set it. Setting `want_nll=True` also gives the original model's teacher-forced NLL on these stays as a bonus baseline. RD is all train stays and shares only 3 with the validation set, which is acceptable. Rerunning with the fix:

```python
code_gen = code_gen.replace("t_hide=1e9,store=store)","t_hide=1e9,want_nll=True,store=store)")
open('/work/gen_wc.py','w').write(code_gen); exec(code_gen)
t0=time.time()
Dw,res_w=gen_wc(RD,armsW,W,a_i,c_i,pf_np,seed=0)
print({k:v.shape for k,v in Dw.items()}, round(time.time()-t0,1))
print('codes', {CNAME[c]:int((Dw['code']==c).sum()) for c in np.unique(Dw['code'])})
print('nll type', type(res_w['nll']), (len(res_w['nll']) if hasattr(res_w['nll'],'__len__') else ''))
```