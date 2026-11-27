## Step 18: add the missing lags

**What the last result showed.** The 12-lag regression failed only because I had built lag columns up to 9.

**What I will do next.** Build lags up to 14 and rerun:
- the 12-lag lab impulse response;
- the 8-lag finger impulse response, as a cross-check.

**What would count as evidence.** The same hypotheses as before. Strong persistence would mean a large cumulative effect. A finger response that matches the lab response would confirm both measure the same hidden glucose.

```python
for m in range(0, 15):
    xs[f'I_l{m}'] = grp.ins_rate.shift(m)
    xs[f'e_l{m}'] = grp.e_ins.shift(m)
    xs[f'lg_l{m}'] = grp.lg.shift(m)
b, se, n = irf(xs, 'glucose_lab', 12)
print("LAB K=12 n=",n)
print("  h_m :", np.round(b,2).tolist())
print("  se  :", np.round(se,2).tolist())
print("  cumsum:", np.round(np.cumsum(b),2).tolist())
hL12 = b.copy(); hL12se = se.copy()
b, se, n = irf(xs, 'glucose_finger', 8)
print("FINGER K=8 n=",n, " h:", np.round(b,2).tolist(), " se:", np.round(se,2).tolist())
```