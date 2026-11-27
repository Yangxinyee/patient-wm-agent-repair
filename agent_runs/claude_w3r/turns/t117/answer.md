The h error is spread over many dimensions (the top 12 carry only 40% of it), and a_bar is moderate everywhere (0.28 to 0.44). So it is not a slow-decay formula error. In some dimensions my value is off by around 10 to 15, even the sign of dimension 135 is wrong, while values run to about 180.

Since out matched better on action bundles, I will check the h relative error by bundle type. If h is accurate on action bundles, the core-step formula is right and the problem is how I encode measurement values into e_k (bins, z-scores or units). If h is equally poor on both types, the problem is in the step itself.

```python
rel_by=[]; 
for k in range(1,len(B0)):
    r_=rows0[k]; hp=torch.tensor(SX[rows0[k-1],192:960].reshape(4,192),dtype=torch.float64)
    zk=torch.tensor(SX[r_,960:],dtype=torch.float64); idx_=list(range(k+1))[-128:]
    Mem=torch.stack([EU0[j][0] for j in idx_]); Mage=torch.tensor([(B0[k][0]-B0[j][0])*3600 for j in idx_],dtype=torch.float64)
    y,o,E=core_step_v(hp,dts[k-1],EU0[k][0],EU0[k][1],zk,Mem,Mage,'std')
    ht=torch.tensor(SX[r_,192:960].reshape(4,192),dtype=torch.float64)
    codes=sorted(set(a['code'] for a in B0[k][1])); kinds=sorted(set(a['kind'] for a in B0[k][1]))
    rel_by.append((k, round(float((y.mean(0)-ht.mean(0)).norm()/ht.mean(0).norm()),3), round(float((o-torch.tensor(SX[r_,:192],dtype=torch.float64)).abs().median()),3), round(dts[k-1]), codes, kinds, len(B0[k][1])))
inv={v_:k_ for k_,v_ in CODE.items()}
for t_ in rel_by[:30]: print(t_[0], 'relh',t_[1],'out',t_[2],'dt',t_[3], [inv.get(c_,c_) for c_ in t_[4]], 'kinds',t_[5],'n',t_[6])
```