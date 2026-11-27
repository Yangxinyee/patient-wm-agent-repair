Each DEC decoder is one multi-output ridge fit (3 lags × 192 inputs = 576 coefficients). I will reshape it to 3×192 and redo the evaluation per lag.

```python
for nm,codes,rw in [('glucose_finger',[0,1],ins_rows),('map_arterial',[2,3],map_rows)]:
    W3=np.asarray(DEC[nm].coef_).reshape(3,192)
    for j in range(3):
        w0=W3[j]; print(nm[:5],'lag',j+1,'noise',round(np.std(Rf[rw]@w0),3),[(k[0][:3]+str(k[1]),resp(w0,k,codes)) for k in keysI+keysN])
```