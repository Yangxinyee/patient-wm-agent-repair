The kernel already holds the model's real states for the 100 selS stays from fwd_2: SX (5301 bundles × 992 numbers), with stay ids, hours and drug flags. It also holds several per-stay structures (obj, key_k, bidx) that probably contain my built bundles.

To check a local core step against these states I need the bundle atoms per stay in the format my encoder uses, lined up with the rows of SX. So next I will look at the structure of obj, key_k and bidx, build an encoder instance, and confirm that bundle counts per stay match the states rows.

- **Confirms:** bundle counts match the SX rows for every stay, so I can feed exact previous states into a local core step.
- **Refutes:** the counts differ. Then I would rebuild bundles with build_bundles first.

```python
s0=selS[0]
for nm_,o_ in [('obj',obj),('key_k',key_k),('bidx',bidx)]:
    v_=o_[s0]; print(nm_, type(v_).__name__, (len(v_) if hasattr(v_,'__len__') else v_))
    if isinstance(v_,(list,tuple)) and len(v_)>0: print('   first:', str(v_[0])[:400])
    elif isinstance(v_,dict): print('   keys:', list(v_.keys())[:10])
    elif isinstance(v_,np.ndarray): print('   ', v_[:10])
print('SX rows for s0:', (SST==s0).sum(), ' n_atoms:', z['n_atoms'][SST==s0][:12], ' hours:', SHR[SST==s0][:12])
```