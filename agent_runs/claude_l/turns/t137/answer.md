With the fixes in place, cuff, arterial and lab glucose atoms all encode exactly; only finger glucose is still off, by about 0.5. Lab glucose uses the same unit and works, so the finger atom's normalization, bins or unit must differ from what I assume.

I'll print the vocab entries for finger and lab glucose and test these variants on single-finger bundles:
- finger using the lab's median, IQR and edges;
- other unit rows;
- dropping the clipping.

```python
for nm in ['glucose_finger','glucose_lab']:
    d=NUMV[(nm,'value')]; print(nm, d['median'], d['iqr'], d['fine_bin_edges'])
fing=[(i,ea[0],err) for i,key,err,ea in pb if len(ea)==1 and ea[0]['code']==6]
def test_f(fn):
    errs=[]
    for i,a,_ in fing:
        a2=fn(dict(a)); e=set_enc(W0,[a2],'encode').detach().numpy(); errs.append(np.abs(Pe@e-recs[i][4]).max())
    return np.round(np.median(errs),4)
def mk_norm(src):
    def f(a):
        x=a['num'][0]*NUMV[('glucose_finger','value')]['iqr']+NUMV[('glucose_finger','value')]['median']
        d=NUMV[(src,'value')]; n=list(a['num']); n[0]=float(np.clip((x-d['median'])/d['iqr'],-10,10)); a['num']=tuple(n)
        a['vb']=1+int(np.searchsorted(np.array(d['fine_bin_edges']),x,side='left')); return a
    return f
print('lab norm', test_f(mk_norm('glucose_lab')))
for u_ in range(6):
    def fu(a,u_=u_): a['unit']=u_; return a
    print('unit',u_, test_f(fu))
def fnob(a): a['vb']=0; return a
print('no vb', test_f(fnob))
def fnoz(a): n=list(a['num']); n[0]=0.0; a['num']=tuple(n); return a
print('no z', test_f(fnoz))
```