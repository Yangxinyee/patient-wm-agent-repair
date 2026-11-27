The pooled readouts from n3 are uneven. The expected rate given the state reads well (band-mean R2 0.889), but the actual insulin rate reads poorly (R2 0.729), so the surprise only reaches R2 0.29. The rate signal is blurred by the action encoder's nonlinearity and the norm3 normalisation.

So I'll split the job. The exact rate and an insulin-present gate will be written into x1 through head 1's value path, which reads the u_k token with weight about 1.0 at every bundle and does not depend on the state. The FFN units will then combine them with the state-based expected rate. That needs low-variance directions in x1, so a small written signal is read precisely and barely disturbs the rest, and a fairly constant std(x1) across bundles. I'll check both at drug and measurement bundles.

```python
r2n = prelude2(TOKn, W_)
x1d = r2['x1'][:, :4, :].reshape(-1,192).numpy(); x1n = r2n['x1'][:, :4, :].reshape(-1,192).numpy()
x1all = np.r_[x1d, x1n]
sd_tok = x1all.std(1); nrm = np.linalg.norm(x1all,axis=1)
print('x1 slot-token: std over dims mean %.3f cv %.3f | norm mean %.2f' % (sd_tok.mean(), sd_tok.std()/sd_tok.mean(), nrm.mean()))
C_ = np.cov((x1all - x1all.mean(1,keepdims=True)).T)
ev, evec = np.linalg.eigh(C_)
print('eig smallest', np.round(ev[:6],5), 'largest', np.round(ev[-3:],2), 'total %.2f' % ev.sum())
w3 = load('core.prelude.0.norm3.weight').numpy(); print('norm3 w |mean| %.3f min %.3f' % (np.abs(w3).mean(), np.abs(w3).min()))
```