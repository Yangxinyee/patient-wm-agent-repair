The pulse response is clear and short-lived. One extra unit/h of insulin for one hour, at the 180 threshold, changes glucose_finger as follows:

| hours after the pulse | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| change (mg/dL) | -2.0 | -1.9 | -1.7 | -1.0 | -0.5 | about 0 |

Later doses barely respond: the protocol feedback is only about -0.05 units/h for a few hours. So most of this is the drug's own time course. At 150 the effect persists longer. But there, crossing also triggers next-hour measurement and more insulin (+0.33 units at h+1), so that threshold is contaminated by the measurement and dosing policy.

For the evaluation, doses are fixed at twice the record, so I need the pure impulse response IR(k) with the protocol feedback removed. Next I will deconvolve jumpGF(k) = sum_j IR(k-j)·jumpI(j) at each threshold, pooling 180 and 220, which are clean because glucose is measured hourly above 160. I will also check the effect on glucose_lab via RD. Similar IR estimates at 180 and 220, and a glucose_lab effect of the same size, would mean the drug acts on true glucose rather than on one measurement.

```python
def rd_series(d, T, w, K=8):
    jy=[];sy=[];ji=[]
    for k in range(1,K+1):
        a,b,_ = rd_jump(d,T,w,f'GF_p{k}'); jy.append(a); sy.append(b)
        c,_,_ = rd_jump(d,T,w,f'I0_p{k-1}'); ji.append(c)
    return np.array(jy), np.array(sy), np.array(ji)
def deconv(jy, ji):
    K=len(jy); IR=np.zeros(K)
    for k in range(K):  # jy[k] (horizon k+1) = sum_{j=0..k} IR[k-j]*ji[j]
        IR[k] = (jy[k] - sum(IR[k-j]*ji[j] for j in range(1,k+1)))/ji[0]
    return IR
res={}
for T in [180,220,260,150]:
    jy,sy,ji = rd_series(d,T,20)
    res[T]=(jy,sy,ji)
    print(T, "IR:", np.round(deconv(jy,ji),2), " sum1..6:", round(deconv(jy,ji)[:6].sum(),2))
# pooled 180+220 (stack with threshold-specific intercepts/slopes): approx by inverse-variance avg of jumps
jy = (res[180][0]/res[180][1]**2 + res[220][0]/res[220][1]**2)/(1/res[180][1]**2+1/res[220][1]**2)
ji = (res[180][2]+res[220][2])/2
print("pooled IR:", np.round(deconv(jy,ji),2))
# glucose_lab RD
for k in range(1,5): P[f'GL_p{k}'] = g['GL'].shift(-k)
d = P[P.h>=0]
for T in [180,220,150]:
    print(T, "GL jumps:", [ (round(rd_jump(d,T,20,f'GL_p{k}')[0],2), round(rd_jump(d,T,20,f'GL_p{k}')[1],2)) for k in range(1,5)])
```