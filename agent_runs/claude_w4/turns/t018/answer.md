## Step 17: the insulin effect builds up and persists for many hours

**What the last result showed.** Using lab glucose (selection-free) and dose surprises as instruments, a one-hour dose of 1 unit/h lowers glucose by:

| hours later | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| effect (mg/dL) | −1.8 | −2.1 | −1.9 | −1.85 | −1.4 | −1.0 | −1.0 | −0.6 |

The SEs are about 0.2. The effect peaks around 2 hours, which fits a short delay since the dose 1 hour before has run only 55 minutes by the reading, then decays slowly. A sustained extra 1 unit/h therefore accumulates to about −11.6 mg/dL after 8 hours, and probably more by 12 hours.

**What this means.** In the evaluation (×2 from hour 12, read over hours 13–24), the true effect will be several times the one-hour effect.

**What I will do next.**
1. Extend to 12 lags (the readout spans 12 hours) to get the tail.
2. Check that the finger-glucose impulse response, restricted to stays with hourly checks, agrees with the lab version.
3. Test linearity: is the response proportional to dose, or saturating? The evaluation also scores ×0, ×0.5, ×1.5 and ×3.

**What would count as evidence.** A good fit of h_m to a smooth kernel (for example a two-compartment exponential) would let me predict all the edits from one structural model.

```python
for K_ in [12]:
    b, se, n = irf(xs, 'glucose_lab', K_)
    print("LAB K=",K_, "n=",n)
    print("  h_m :", np.round(b,2).tolist())
    print("  se  :", np.round(se,2).tolist())
    print("  cumsum:", np.round(np.cumsum(b),2).tolist())
hL12 = b.copy()
# finger outcome but restricted: within stays where all latest glucose lags >150 (hourly checks)... instead just use finger and compare
b, se, n = irf(xs, 'glucose_finger', 8)
print("FINGER K=8 n=",n, " h:", np.round(b,2).tolist(), " se:", np.round(se,2).tolist())
# nonlinearity test: one-step with lab outcome, by rate level within band 2 using dummies of rate at s-1 (instrumented implicitly: within band, rate variation is noise)
d = xs[xs.glucose_lab.notna() & xs.lg_l1.notna() & (xs.age_glucose_finger.shift(1)==0)].copy()
```