Bundle types are now clear. Drug bundles at h + 0.083 number 1,619: 666 insulin only, 473 NE only, 358 both, and 122 with only END atoms. The has_insulin flag matches recorded insulin exactly.

Next I will examine the z posterior mean at insulin-only bundles (excluding NE to avoid mixing). I will regress each of the 32 z dimensions on the dose with controls for the reading at that hour and its tier. That gives:
1. the dose direction in z and how much variance it explains;
2. whether z also depends on the tier base at a fixed dose. If so, z encodes the dose relative to expectation, the deviation code I want; if not, it encodes the raw level.

I will do the same on the change in out and in the h slots, comparing after the drug bundle with the preceding reading bundle.

```python
X = S['x']
st2['row'] = np.arange(len(st2))
st2['tf'] = tier(st2.glucose_finger)
ib = st2[(st2['ins']) & (~st2['ne']) & st2.glucose_finger.notna()].copy()
ib = ib[~ib.glucose_finger.round().isin([150,180,220,260]) & (ib.tf>=1)]
print("insulin-only bundles with finger:", len(ib))
Z = X[ib.row.values, 960:992]
prev_rows = ib.row.values - 1
assert (st2.frac.values[prev_rows]==0.0).all()
dOut = X[ib.row.values, :192] - X[prev_rows, :192]
dH = X[ib.row.values, 192:960] - X[prev_rows, 192:960]
g = ib.glucose_finger.values; D = ib.insulin.values
base = ib.tf.values.astype(float); base[base==4]=4.2
def multi_ols(Y, Xc):
    Xm = np.column_stack(Xc); B, *_ = lstsq(Xm, Y, rcond=None); R = Y - Xm@B
    return B, R
ctrl = [np.ones(len(ib)), g, g**2/100] + [(ib.tf.values==t).astype(float) for t in [2,3,4]]
for name, Y in [('z',Z),('dOut',dOut),('dH',dH)]:
    B0, R0 = multi_ols(Y, ctrl)
    B1, R1 = multi_ols(Y, ctrl+[D])
    frac = 1 - (R1**2).sum()/ (R0**2).sum()
    bD = B1[-1]
    print(name, "frac var explained by D beyond ctrl: %.3f"%frac, " |bD|=%.3f"%np.linalg.norm(bD), " resid rms per dim %.3f"%np.sqrt((R1**2).mean()))
```