The oracle shows what run 207 did. In my local rollout, at every glucose head, `make_oracle` tilts the predicted bin distribution so its mean moves by r = Σ_k G_k·(dose_arm − dose_recorded) at lags k. `truth_rows` gives the true effect from the record-based kernel K.

So a direct one-step correction with lag-1 gain −1.5 is propagated by the model's own dynamics to about 80% of the truth. This holds across x2, x1/2, x0, x3 and the hour-16 start. The arms are consistent with each other, so the model's propagation is close to linear in the injected shift. The remaining gap is mainly at the middle lags.

The notes also confirm that I had already implemented a deviation-based oracle, `make_dev_oracle`. It uses the dose minus the policy dose p(glucose), which is the version that should keep factual fit intact.

The notebook does not record what runs 149–206 produced (dec.py, evalh.py, states_ins.py, tf.py, wI_chan.npy, wI_head2.npy, transfer_T.npy). I will print those small files and list the kernel's live variables. Then I can reuse the decoders or channel vectors I already computed instead of rebuilding them. Again this restores context rather than testing a hypothesis.

```python
for f in ['dec.py','evalh.py','states_ins.py','tf.py','policy.py']:
    print('=====',f,'====='); print(open('/work/'+f).read())
import types
vs=[]; fs=[]
for k,v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v,types.ModuleType): continue
    if callable(v) and not isinstance(v,(np.ndarray,torch.Tensor)): fs.append(k); continue
    if isinstance(v,(np.ndarray,torch.Tensor)): vs.append('%s%s'%(k,tuple(v.shape)))
    elif isinstance(v,(pd.DataFrame,)): vs.append('%s[df %s]'%(k,v.shape))
    elif isinstance(v,(dict,list,tuple)): vs.append('%s[%s %d]'%(k,type(v).__name__,len(v)))
    elif isinstance(v,(int,float,str)): vs.append('%s=%s'%(k,str(v)[:30]))
print('VARS:',' '.join(vs)[:5000]); print('FUNCS:',' '.join(fs)[:3000])
```