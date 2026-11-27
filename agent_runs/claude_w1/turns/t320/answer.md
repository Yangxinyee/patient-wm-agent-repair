The single invariant direction g = 1/γ does not separate the groups. Code means are −2.6 (finger), −0.05 (lab), −4.1 (art) and −3.1 (cuff), with spreads of about 2 inside each code, so they overlap heavily. A z-proof gate therefore needs more than that one direction.

Next I try a Fisher/LDA gate: the direction that best separates glucose codes from MAP codes relative to within-code variation in the replica head inputs. It will be constrained to be orthogonal to the subspace the z-noise reaches in q, which is diag(γ)·centring·z_head.weight (32 dims, first order). The gate is then insensitive to z draws by construction, and can still use directions where the state varies little.

I'll also check:
- the signatures of my helper functions and the structure of the stored replica edit responses (EDR), which I need to rebuild the decoders;
- how much of the current decoders lies in the z subspace.

If the LDA gate gives a margin of many within-code standard deviations between the groups, v2 is feasible. If the groups still overlap, I fall back to a no-gate design or to the original model.

```python
import inspect
for fname in ['run_rows', 'run_rows_wc', 'rows_of', 'q_rows', 'make', 'build', 'mvdr_dec', 'solve_unit', 'diag', 'cellmean',
              'head_logits', 'nll_from', 'probs', 'LO', 'gen_rows_all', 'proc_stay', 'head_mean', 'within_sd', 'get_w', 'score', 'report']:
    if fname in globals():
        try:
            print(fname, str(inspect.signature(globals()[fname]))[:110])
        except Exception:
            print(fname, 'sig?')
print('EDR keys', list(EDR.keys()))
k0 = list(EDR.keys())[0]
print({kk: (getattr(vv, 'shape', None) if hasattr(vv, 'shape') else type(vv).__name__) for kk, vv in EDR[k0].items()})
X64 = Xf.astype(np.float64)
mu_c = np.stack([X64[cf == c].mean(0) for c in range(4)])
Sw = sum(np.cov(X64[cf == c].T) * (cf == c).sum() for c in range(4)) / len(cf)
Wz = np.asarray(load_original('z_head.weight'), dtype=np.float64)
Pc = np.eye(192) - 1.0 / 192
Zs = gam[:, None] * (Pc @ Wz)
Qz, _ = np.linalg.qr(Zs)
Pz = np.eye(192) - Qz @ Qz.T
Dg = (mu_c[0] + mu_c[1]) / 2 - (mu_c[2] + mu_c[3]) / 2
for lamf in [1e-5, 1e-4, 1e-3]:
    lam = lamf * np.trace(Sw) / 192
    wg = Pz @ np.linalg.solve(Pz @ Sw @ Pz + lam * np.eye(192), Pz @ Dg)
    tg = X64 @ wg
    st = []
    for c in range(4):
        s = tg[cf == c]
        st.append((s.mean(), s.std(), np.percentile(s, .01), np.percentile(s, 99.99)))
    lo_on = min(st[0][2], st[1][2]); hi_off = max(st[2][3], st[3][3]); sdw = max(x[1] for x in st)
    print('lamf %g: code mean/sd ' % lamf + ' | '.join('%.3f/%.3f' % (a, b) for a, b, _, _ in st),
          ' margin(min on - max off)=%.3f  sd_max=%.4f  ratio=%.1f' % (lo_on - hi_off, sdw, (lo_on - hi_off) / sdw))
for key in [('ins', 'mvdr', -2.8), ('ne', 'mvdr', 3)]:
    v = DEC2[key].astype(np.float64)
    print(key, 'norm %.3f  frac in z-subspace %.3f' % (np.linalg.norm(v), np.linalg.norm(Qz.T @ v) / np.linalg.norm(v)))
```