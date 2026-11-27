The keys are (stay_id, minute) tuples. My earlier int lookups had silently returned empty atom sets, which is why every z check failed. enc2.py also reminded me what the exact encoder does: atoms are embedded, set-pooled, then MLP + LayerNorm, and `zpost2` returns (64 outputs, 192 hidden). This run re-encodes every bundle with the correct key and repeats the validation of z and `core_step` against the tool states. I expect relative errors of about 1e-6 or less. Anything larger would point to a remaining convention problem in `core_step`'s inputs.

```python
EA=torch.zeros(N3,192); UA=torch.zeros(N3,192)
for k in range(N3):
    e,u=eu2(atoms_by_bundle[(int(sid[k]),int(round(hr[k]*60)))]); EA[k]=e.float(); UA[k]=u.float()
ez=[];ez64=[];eh=[];eo=[]
for k in ksv:
    k=int(k); ctx,z,h,o=step_local(k); zt=X3f[k,960:]; ht=X3f[k,192:960]; ot=X3f[k,:192]
    ez.append(float((z.reshape(-1)-zt).norm()/zt.norm())); eh.append(float((h.reshape(-1)-ht).norm()/ht.norm())); eo.append(float((o.reshape(-1)-ot).norm()/ot.norm()))
    o_p,h_p=prev64(k); c64=ctx_of_state(o_p[None],h_p[None],np.array([dtl[k]])).reshape(-1)
    z64=zpost2(c64[None],EA[k].double()[None],UA[k].double()[None])[0].reshape(-1)[:32]; ez64.append(float((z64-X64[k,960:]).norm()/X64[k,960:].norm()))
pc=lambda a: np.round(np.percentile(a,[50,90,100]),6)
print('z32',pc(ez),' z64',pc(ez64),' h',pc(eh),' out',pc(eo))
```