Decaying h with dt in seconds is clearly right: removing the decay or changing its units triples the error. But the per-slot error on h is large (median 2.3) even though `out`, the normalised mean over slots, matches to 0.028. Either h has large magnitudes, so 2.3 is small in relative terms, or my slots differ from the model's (for example permuted, or carrying a large common offset that out_norm removes).

Next I will compare my slots with the model's: magnitudes, relative error per slot, error after removing the slot mean, and whether a slot permutation fits better.

- **Small relative error:** the local core is good enough to design and test an edit.
- **Large relative error:** something in the slot update is wrong, and the out match is only partial.

```python
k=10; r_=rows0[k]; hp=torch.tensor(SX[rows0[k-1],192:960].reshape(4,192),dtype=torch.float64)
zk=torch.tensor(SX[r_,960:],dtype=torch.float64); idx_=list(range(k+1))[-128:]
Mem=torch.stack([EU0[j][0] for j in idx_]); Mage=torch.tensor([(B0[k][0]-B0[j][0])*3600 for j in idx_],dtype=torch.float64)
y,o,E=core_step_v(hp,dts[k-1],EU0[k][0],EU0[k][1],zk,Mem,Mage,'std')
ht=torch.tensor(SX[r_,192:960].reshape(4,192),dtype=torch.float64)
print('norm per slot true', ht.norm(dim=1).numpy().round(1), ' mine', y.norm(dim=1).numpy().round(1))
print('rel err per slot', ((y-ht).norm(dim=1)/ht.norm(dim=1)).numpy().round(3))
C=np.array([[float(((y[i]-ht[j]).norm()/ht[j].norm())) for j in range(4)] for i in range(4)]); print('cross rel err (mine i vs true j)\n',C.round(3))
yc=y-y.mean(0); hc=ht-ht.mean(0)
print('mean-slot rel err', float((y.mean(0)-ht.mean(0)).norm()/ht.mean(0).norm()), ' centred rel err', float((yc-hc).norm()/hc.norm()))
print('abs max true h', float(ht.abs().max()), ' median |h|', float(ht.abs().median()))
```