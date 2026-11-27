This is a striking finding. In the model's actual posterior, dimensions 9 and 20 at insulin bundles are almost binary. They are +1.08 / +0.84 for rates of exactly 1.0 or 1.5 u/h, and −1.1 / −1.3 for 0.5 u/h and for every rate ≥ 2, with within-group sd of only about 0.01. So the model feeds in some feature of the rate that I am not reproducing. My computation gives the "1.0/1.5" value even for 0.5, and changing the bins did not move it.

Next I brute-force the numeric inputs z_rate and z_dose of the 0.5 u/h atom in bundle 9 over a grid, and check which values reproduce the actual posterior.
- If a value other than my z = −1 fits, the model's normalisation of drug rates differs from mine.
- If no value fits, the missing feature is something else.

```python
def mod_num(zr, zd):
    def f(a,v):
        if a['code']==8 and a['kind']==0:
            num_old=torch.zeros(6); num_old[2]=zsc('insulin','dose',0.5); num_old[3]=1; num_old[4]=zsc('insulin','rate',0.5); num_old[5]=1
            num_new=num_old.clone(); num_new[2]=zd; num_new[4]=zr
            v=v-Wo['embed.numeric.weight']@num_old+Wo['embed.numeric.weight']@num_new
        return v
    return f
zt9=torch.tensor(X0[9,960:])
gr=np.linspace(-10,10,41); resg=[]
for zr in gr:
    for zd in gr:
        zp=zk_custom(9,mod_num(float(zr),float(zd))); resg.append((round(float((zp-zt9).abs().mean()),4),zr,zd,round(float(zp[9]),2),round(float(zp[20]),2)))
resg.sort(); print(resg[:6]); print('actual dims 9,20:', float(zt9[9]), float(zt9[20]))
```