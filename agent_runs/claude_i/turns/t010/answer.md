The full insulin policy is now clear:
- If there is a finger reading at hour h, insulin is given when it is ≥150.
- Otherwise, if there is a lab reading, insulin is given when that is ≥150 (99.5% agreement).
- Otherwise no insulin is given.
- Rate = band base (1, 2, 3, …) × multiplicative noise, rounded to 0.5.

Two checks support treating the noise as random. Its autocorrelation across consecutive dosing hours is 0.005, and the variance of per-stay mean deviations matches what iid noise predicts. So the rate scatter is plausibly a randomised experiment.

Finger and lab readings agree with no bias (difference SD about 9, growing with level, i.e. about 3.7% multiplicative noise each).

Next is the first causal estimate, using two independent designs:
- **Within band:** regress the next-hour glucose on the random rate, controlling for the current reading.
- **Regression discontinuity:** at 150, readings of 140–149 get no insulin while 151–160 get about 1 u/h.

If the two designs agree, that supports a causal reading. A slope near zero would point to a small or absent effect.

```python
Q = P.copy()
Q['gf_next'] = Q.groupby(level='stay_id').gf.shift(-1)
Q['gl_next'] = Q.groupby(level='stay_id').gl.shift(-1)
Q['g_next'] = Q.gf_next.combine_first(Q.gl_next)
Q['gdec'] = Q.gf.combine_first(Q.gl)
Q = Q[Q.index.get_level_values('h')>=0]
import statsmodels.api as sm
```