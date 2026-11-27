std(x1) varies 17% across slot tokens, so anything read from n3 is scaled by about ±17%. The x1 spectrum has many directions with almost no natural variance (sd about 0.015, against token norms of about 800). Those can serve as a private carrier: a signal written there barely disturbs anything and can be read back precisely.

The plan: head 1's value path writes the exact insulin rate into such a carrier (it doubles as the on/off gate). Gated FFN unit pairs then output δ × (rate − expected rate from state) along a direction δ chosen to produce a glucose-belief change in the next state.

Next I'll check how much of the std variation is between slots rather than between bundles. Then I'll fit an exact linear readout of the insulin rate from LN1(u_k) (the norm1 of the u token): zero for no-action and NE-only tokens, the insulin rate otherwise.

```python
sdd = r2['x1'][:, :4, :].numpy().std(2); sdn = r2n['x1'][:, :4, :].numpy().std(2)
print('x1 std per slot drug mean', np.round(sdd.mean(0),2), 'cv', np.round(sdd.std(0)/sdd.mean(0),3), '| meas mean', np.round(sdn.mean(0),2), 'cv', np.round(sdn.std(0)/sdn.mean(0),3))
w1n, b1n = load('core.prelude.0.norm1.weight'), load('core.prelude.0.norm1.bias')
L1 = lambda x: torch.nn.functional.layer_norm(x, (192,), w1n, b1n, 1e-5)
Ud_ = L1(TOK[:,5,:]).numpy(); Un_ = L1(TOKn[:,5,:]).numpy()
print('u_const spread at meas bundles %.2e' % np.abs(Un_ - Un_[0]).max())
uc = Un_[0]
A_ = np.r_[Ud_, uc[None]]; t_ = np.r_[rI, 0.0]
P_ = np.eye(192) - np.outer(uc, uc)/uc.dot(uc)   # enforce b.uc = 0
Ap = A_ @ P_
bI = np.linalg.lstsq(Ap, t_, rcond=None)[0]; bI = P_ @ bI
err = A_ @ bI - t_
iN = hasN_d==1
for nm_, m_ in [('ins-only', iI & ~iN), ('combo', iI & iN), ('NE-only', ~iI & iN), ('none', ~iI & ~iN)]:
    e_ = err[:-1][m_]; print(nm_, m_.sum(), 'max|err| %.4f rms %.4f' % (np.abs(e_).max() if len(e_) else 0, np.sqrt((e_**2).mean()) if len(e_) else 0))
print('u_const err %.2e' % err[-1], '|bI| %.3f' % np.linalg.norm(bI))
```