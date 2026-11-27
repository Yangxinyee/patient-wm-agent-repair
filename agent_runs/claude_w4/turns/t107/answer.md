Restricted to rates the encoder can see, the insulin rate is exactly decodable from u_k for insulin-only bundles: rms error 0.0002 with a modest weight norm of about 300. Mixed bundles are harder, with rms 0.066 and worst case 0.32 for a single readout.

What matters for the dose readouts is the error in the change of the readout when the rate is multiplied while the NE rate stays fixed. I now fit one insulin-rate readout on insulin-only plus mixed bundles, and one NE-rate readout on NE-only plus mixed bundles. I then measure the errors in those changes on realistic recorded rates and on the multipliers the evaluation uses.

If the change errors stay below about 0.05 units/h (insulin) and 0.01 (NE), a single gated readout per drug is enough. Otherwise I need separate readouts for single-drug and mixed bundles, plus a gate to choose between them.

```python
mI = np.isin(aC, rI0)
UI_all = np.vstack([Ui_only, UC]); yI_all = np.r_[rI0, aC]
fI = Ridge(alpha=1e-10).fit(UI_all, yI_all)
rN0 = np.unique(np.r_[0.0, np.arange(0.005, 3.0001, 0.005)])
Un_only = np.stack([U_of([ne_atom(float(r))]) for r in rN0])
UN_all = np.vstack([Un_only, UC]); yN_all = np.r_[rN0, bC]
fN = Ridge(alpha=1e-10).fit(UN_all, yN_all)
rec_I = np.arange(0.5, 6.01, 0.5); rec_N = np.r_[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
def fIu(a, b=None): return fI.predict(U_of([ins_atom(float(a))] + ([ne_atom(float(b))] if b is not None else []))[None])[0]
def fNu(b, a=None): return fN.predict(U_of(([ins_atom(float(a))] if a is not None else []) + [ne_atom(float(b))])[None])[0]
for m in [0, 0.5, 1.5, 2, 3]:
    eo = [fIu(min(m*a, 11.5)) - fIu(a) - (min(m*a,11.5) - a) for a in rec_I]
    ec = [fIu(min(m*a,11.5), b) - fIu(a, b) - (min(m*a,11.5) - a) for a in rec_I for b in rec_N]
    eno = [fNu(m*b) - fNu(b) - (m-1)*b for b in rec_N]
    enc = [fNu(m*b, a) - fNu(b, a) - (m-1)*b for b in rec_N for a in rec_I]
    print('m=%.1f  dI err: only max %.3f | comb rms %.3f max %.3f   dN err: only max %.4f | comb rms %.4f max %.4f' % (m, np.abs(eo).max(), np.sqrt(np.mean(np.square(ec))), np.abs(ec).max(), np.abs(eno).max(), np.sqrt(np.mean(np.square(enc))), np.abs(enc).max()))
```