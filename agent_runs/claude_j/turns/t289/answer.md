The cost blow-up comes from projecting the readout off the last_numeric columns. With that projection, insulin's out-of-fold cost at gain 1 is 0.09, against 0.0018 for the otherwise identical unprojected fit in run 249. The readout evidently needs those directions to cancel its dependence on the last glucose value within a bin.

The weak-unit scan has good news. Units 118, 158 and 149 (≤0.021 logits), as well as 89 and 170 (≤0.003), are near-dead. So each drug can get its own gate-cancelling unit:
- insulin: 89, cancelled by 149;
- drug B: 170, cancelled by 118.

Next run: refit with no projection, and with projection only off the last_numeric columns unused for measurements (1, 2, 4, 5, 7). I also report the readout's spread over glucose rows and over MAP rows. The unprojected fit should reproduce run 249: fraction about 0.46 at cost about 0.002 for insulin, and about 0.31 for drug B.

```python
used=[0,3,6]; dcols=[i for i in range(8) if i not in used and np.linalg.norm(LNW[:,i])>1e-6]
Ld=LNW[:,dcols]; Pd=np.eye(192)-Ld@np.linalg.pinv(Ld) if len(dcols) else np.eye(192); print('drug-only LN cols',dcols)
for nm,fa_,c_,v_,rho,lam in [('INS',fac,cf,vG,0.4,30),('NE',faN,cN,vM,0.3,300)]:
    for pn,P_ in [('I',np.eye(192)),('Pd',Pd)]:
        w,fr,cst=fitw2(fa_,c_,v_,rho,lam,0.0,P_,1e-2); r=Xm@w; sr=np.std(Xw@w)
        print(nm,pn,'frac',round(float(fr),3),'cost',cst,'R glu/sr',round(np.ptp(r[ipos])/sr,1),'R map/sr',round(np.ptp(r[~ipos])/sr,1),'code dots/sr',np.round(EC[[5,6,7,10,8,9]]@w/sr,1).tolist())
```