Three findings from the last result:
- `WT` is a plain dictionary of weight tensors, so I can test a changed z_in column locally.
- `Jtok` shows the z-token route is weak and badly mixed. The average token Jacobians for glucose and MAP point almost the same way (cosine 0.87–0.94), at about 0.27 mg/dL per unit of token norm.
- Separating glucose from MAP leaves only about 0.13 mg/dL per unit, so a −4 mg/dL shift would need a token change of about 30. That is far outside the linear range.

So injecting the dose effect directly through the z token cannot work.

**Another route:** the notebook says the glucose value head reads an unexplained glucose-specific input, column 0 of `last_numeric`, which may be insulin-related. If the head reads a dose or insulin feature straight from the record, that is a strong path from dose to prediction that bypasses the state.

**This step:** pull short excerpts of ARCHITECTURE.md about the head inputs (last_numeric, status, rate bin, view).
- If one feature carries the insulin dose or time since the last insulin, I will build the edit around it.
- If not, I fall back to a two-stage design: a weak but clean write into the state, then an amplified read-out through z_prior/z_head.

```python
import re
A = open('/model/ARCHITECTURE.md').read().splitlines()
pat = re.compile(r'last|numeric|status|rate.?bin|view|z_head|z_prior|value_bin|transition', re.I)
hits = [i for i,l in enumerate(A) if pat.search(l)]
print(len(A), 'lines;', len(hits), 'hits')
shown = set(); out=[]
for i in hits:
    for j in (i,):
        if j not in shown:
            shown.add(j); out.append(f'{j}: {A[j].strip()[:150]}')
print('\n'.join(out[:45]))
```