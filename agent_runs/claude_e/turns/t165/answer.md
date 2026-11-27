The k-profile explains the insulin shortfall. One hour after the first edited dose the edited model gives -3.4 against a truth of -3.0, so the readout's gain is fine. After that the model adds only about 1 unit per hour while the truth adds 3-4. The gap comes from the lagged effect of earlier doses, mainly the lag-2 effect (the true one-step coefficient at lag 2 is -1.23, the model's +0.13). The edit has no lag-2 term because the lag-2 dose deviation can't be read from the state (R2 about 0.05).

The fix is therefore either a larger lag-1 coefficient (about -2.7 by linear scaling), which costs factual NLL, or cheaper corrections elsewhere to leave room in the budget.

Next I'll build the NE level-readout pair as an edit function gated4. It takes gated3(a_ins, 0) and overwrites the NE unit with the new rne1 readout plus the existing map_arterial gate. I'll check that the pair output equals the readout on map_arterial rows and stays near 0 elsewhere across all 16 draws.

```python
w_ne, b_ne = fit_lin(tm7, y_ne, off, alpha=1e-4, ndr=4)
gain_ne = 0.588
w_neu = w_ne/gain_ne; b_neu = b_ne/gain_ne
W0o = ORIG['W0'] if isinstance(ORIG['W0'], np.ndarray) else npv(ORIG['W0'])
E0 = gated3(-1.9, 0.0)
W0x = (E0['W0'] if isinstance(E0['W0'], np.ndarray) else npv(E0['W0'])).astype(np.float64)
b0x = (E0['b0'] if isinstance(E0['b0'], np.ndarray) else npv(E0['b0'])).astype(np.float64)
gw_n = W0x[u4].copy(); gb_n = b0x[u4]
G = np.einsum('dnk,k->dn', WCall, gw_n.astype(np.float32)) + gb_n
R = rhat_all(w_neu, b_neu)
gelu = lambda x: 0.5*x*(1+scipy.special.erf(x/np.sqrt(2))) if False else torch.nn.functional.gelu(torch.as_tensor(x)).numpy()
import scipy.special
P = gelu(R+G) - gelu(G)
t = np.broadcast_to(tm7, P.shape)
print('target rows: gate min %.1f pct1 %.1f | max|P-R| %.3f  mean|P-R| %.4f' % (G[t].min(), np.percentile(G[t],1), np.abs(P-R)[t].max(), np.abs(P-R)[t].mean()))
print('off rows: G max %.1f | R range %.1f..%.1f | max|P| %.4f' % (G[~t].max(), R[~t].min(), R[~t].max(), np.abs(P[~t]).max()))
print('readout on target: mean %.3f sd %.3f, y mean %.3f; corr %.3f' % (R[t].mean(), R[t].std(), y_ne[tm7].mean(), np.corrcoef(R.mean(0)[tm7], y_ne[tm7])[0,1]))
```