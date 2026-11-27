The measurement policy is simple and creates a known selection problem.

- **Glucose_finger:** after a reading above 150 it is always measured the next hour. At or below 150 it is measured at random about 30% of the time.
- **Glucose_lab:** drawn at random 20% of the time, independent of glucose.

Per-unit effects are similar across bands 1, 2 and 3, about -2.0 at 1 hour and -2.3 at 2 hours, so the effect is linear and homogeneous. Band 1 is slightly weaker at longer lags. That is exactly what selection would produce: extra insulin pushes glucose below 150, after which glucose_finger is measured less often, and only on a schedule that depends on the earlier, lowered value.

Since glucose_lab is measured completely at random, it gives a selection-free estimate. Next I will rerun the surprise distributed-lag regression with glucose_lab at horizons 1 to 10 as the outcome, deconvolve it, and compare it with the glucose_finger version. I will also compare glucose_finger and glucose_lab drawn in the same hour, to learn the offset and noise between the two measurements. A glucose_lab impulse response matching or slightly exceeding the glucose_finger one at long lags would confirm that the true kernel is about -2·k·0.6^(k-1).

```python
for k in range(5,13): P[f'GL_p{k}'] = g['GL'].shift(-k)
dd = P[P.u.notna()].copy()
for b in [1,2,3,4,6]: dd[f'b{b}'] = (dd.base==b).astype(float)
TYL=[]; SEL=[]
for k in range(1,13):
    a,se,n = surprise_reg(dd, f'GL_p{k}')
    TYL.append(a); SEL.append(se)
TYL=np.array(TYL)
print("GL tot:", np.round(TYL,2)); print("se:", np.round(SEL,2))
IRL = deconv(TYL, np.array(TI))
print("GL deconv IR:", np.round(IRL,2), " cum:", np.round(np.cumsum(IRL),2))
both = P[P.GF.notna()&P.GL.notna()]
print("GF-GL same hour: mean", (both.GF-both.GL).mean(), "sd", (both.GF-both.GL).std(), "corr", both[['GF','GL']].corr().iloc[0,1])
print("model kernel -2*k*0.6^(k-1):", np.round([-2*k*0.6**(k-1) for k in range(1,13)],2))
```