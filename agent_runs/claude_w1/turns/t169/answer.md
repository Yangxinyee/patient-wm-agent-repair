The sweep gives clean linear responses. A unit-norm head-2 direction can move finger-glucose predictions by at most about 0.034 mg/dL per u/h at lag 1. Persistence to lags 2–4 is already small (norms 0.008/0.005/0.007), so this injection acts mostly on the next reading, as hoped. NE is harder: its best MAP lag-1 response is 0.024 per unit with noticeable lag-2 persistence (0.018), and NE injections leak strongly into glucose (0.10).

Next, design the directions by projection:
- **Insulin:** maximise the finger lag-1 response while nulling finger lags 2–4 and all MAP responses.
- **NE:** maximise the MAP lag-1 response while nulling MAP lags 2–4 and all finger responses.

I check how much response survives the constraints, how consistent the response is across queries (R² of the lag model), and how big the residual-stream injection would be at full strength. Surviving responses of a few hundredths, with high consistency, would make a scaled-up edit feasible.

```python
def design(target, cons):
    C = np.column_stack(cons); Q, _ = np.linalg.qr(C); v = target - Q@(Q.T@target); a = v/np.linalg.norm(v); return a, target@a
aI, gI = design(KF[:,0], [KF[:,1],KF[:,2],KF[:,3]] + [KA[:,k] for k in range(4)])
aN, gN = design(KA[:,4], [KA[:,5],KA[:,6],KA[:,7]] + [KF[:,k] for k in range(4,8)])
print('insulin: lag1 finger resp per unit %.4f (unconstrained %.4f); NE: lag1 art resp per unit %.4f (unconstrained %.4f)' % (gI, np.linalg.norm(KF[:,0]), gN, np.linalg.norm(KA[:,4])))
for nm, a in [('aI', aI), ('aN', aN)]:
    dF = DM[:,fm].T@a; dA = DM[:,am].T@a
    bF = np.linalg.lstsq(XF, dF, rcond=None)[0]; bA = np.linalg.lstsq(XA, dA, rcond=None)[0]
    r2F = 1-((dF-XF@bF)**2).sum()/max((dF**2).sum(),1e-12); r2A = 1-((dA-XA@bA)**2).sum()/max((dA**2).sum(),1e-12)
    print(nm, 'finger prof', np.round(bF,4), 'R2 %.3f' % r2F); print('   art prof', np.round(bA,4), 'R2 %.3f' % r2A, '| resid inj norm per unit %.3f' % np.linalg.norm(Wout[:,96:144]@a))
```