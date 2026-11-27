The insulin injection direction works for moderate shifts:

- At s = ±5 it moves glucose means by -5.1 / +4.5 (row-to-row sd about 1.1). MAP rows move only about ±0.1 on average (sd 0.3).
- At ±15 the response gets uneven (sd 6) and saturates on the positive side (+10.9).

The NE direction is much worse. It gives only about 0.8 mmHg per unit, and at s = ±5 it leaks into glucose predictions with a row sd of 2 to 3 mg/dL (12 at s = 15). That would add a lot of factual noise to glucose wherever the NE signal is nonzero.

So I need tighter gating. I will try stronger penalties on the off-family response (λ up to 1000) and weaker ridge (ρ). For each setting I report the on-family gain and off-family leak at s = ±5 and ±10. Later I will also train the decoders to output zero on the other family's rows, which gives a second layer of gating.

```python
def test_u(u, on, off, svals=(-10,-5,5,10)):
    res=[]
    for s in svals:
        don = mean_from_A(A[on]+s*u, on)-mA[on]; dof = mean_from_A(A[off]+s*u, off)-mA[off]
        res.append((s, round(float(don.mean()),2), round(float(don.std()),2), round(float(np.sqrt((dof**2).mean())),2)))
    return res
for lam in (100, 1000, 10000):
    for rho in (1e-3, 1e-5):
        ui = solve_u(rg, rm, lam, rho); un = solve_u(rm, rg, lam, rho)
        print(lam, rho, 'ins', round(float(np.linalg.norm(ui)),1), test_u(ui, rg, rm), 'ne', round(float(np.linalg.norm(un)),1), test_u(un, rm, rg))
```