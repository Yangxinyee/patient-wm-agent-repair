The check passed (`Xu` rows equal `n1u` of their rates), but `n1u` cannot take `None`. It calls `float()` on both rates. So "drug absent" is presumably written as rate 0. I will verify that on a recorded token where only one drug is present, and build the no-action token directly as `n1_of(zeros)` (the norm1 bias, as noted earlier). Then I rerun the same linearity test of the constrained rate reads over the extended rate grid.

```python
j1=int(np.where((Yi>0)&(Yn==0))[0][0]); j2=int(np.where((Yi==0)&(Yn>0))[0][0])
print('zero=absent check', np.abs(np.asarray(n1u(Yi[j1],0.0)).reshape(-1)-Xu[j1]).max(), np.abs(np.asarray(n1u(0.0,Yn[j2])).reshape(-1)-Xu[j2]).max())
u0=np.asarray(n1_of(np.zeros(192))).reshape(1,-1)
Xn_=np.vstack([Xoth,u0.repeat(200,0)])
Gi=np.stack([np.asarray(n1u(float(r),0.0)).reshape(-1) for r in rg])
Gn=np.stack([np.asarray(n1u(0.0,float(r)/4)).reshape(-1) for r in rg])
for nm,yv_,G,sc in [('ins',Yi,Gi,1),('ne',Yn,Gn,4)]:
    for lamn in [0.1,1,10]:
        w=rread(Xu,yv_,Xn_,lamn)
        print(nm,lamn,'res',np.round((Xu@w-yv_).std(),3),'negmax',np.round(np.abs(Xoth@w).max(),3),'u0',np.round((u0@w)[0],3),
              'grid',np.round(G@w*sc,2))
```