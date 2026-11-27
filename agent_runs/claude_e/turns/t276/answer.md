Offline on the sel1 rows (drug-heavy), the head reproduces the tool closely: predicted-mean correlation .9998, NLL correlation .988. It also honours an output bias `b2`. Main numbers:

- `ins_only(-2.7)` costs .0038 per reading here. Most of it is on glucose_finger (.0112/row).
- The NE unit pair at a_ne=0 is about free. It even lowers the total slightly to .0035.
- The NE gain adds a roughly quadratic cost: +.0006 at a_ne=1, +.0027 at 2, +.0063 at 3.
- The earlier tool-to-sel1 ratio for the NE part was about 0.2. On random stays that would be about +.0001, +.0005 and +.0013.

So a moderate NE gain (1.5–2) seems affordable if I can recover a little elsewhere. One source is the cost of having overwritten units 26/41 (and 131/159): their original output contribution is lost.

Next I will measure that repurposing cost directly with `ins_only(0)` and `gated3(0,0)`. I will also test compensation: a ridge regression, fitted on even-numbered stays, of the lost units' original logit contribution on the 188 untouched units plus a constant. The result is folded into W2 and b2 and scored on the held-out odd stays.

If compensation recovers a meaningful part of the repurposing cost out of sample, it buys budget for the NE gain. If not, I will judge the NE gain on its own cost.

```python
Hn = H_o.numpy() if hasattr(H_o,'numpy') else np.asarray(H_o)
W2o = orig['W2'].numpy(); b2o = load_original('attr_heads.value_bin.2.bias').double().numpy()
stays = R1['stay'].values; te = (stays % 2 == 1); trm = ~te
def costm(Wd):
    _, nl, _, _ = head_run(X, R1, Wd); d = nl - nll_o
    return d, {e: round(d[(ev==e)&te].mean(),4) for e in np.unique(ev)}
def comp_fit(lost, keep, lam=10.0):
    tg = Hn[:, :, lost] @ W2o[:, lost].T; tg = tg - tg.mean(-1, keepdims=True)
    a = Hn[:, trm][:, :, keep].reshape(-1, len(keep)); y = tg[:, trm].reshape(-1, 17)
    a1 = np.hstack([a, np.ones((len(a),1))]); G = a1.T@a1 + lam*np.eye(a1.shape[1]); G[-1,-1] -= lam
    beta = np.linalg.solve(G, a1.T@y)
    at = np.hstack([Hn[:, te][:, :, keep].reshape(-1,len(keep)), np.ones((16*te.sum(),1))]); yt = tg[:, te].reshape(-1,17)
    r2 = 1 - ((yt - at@beta)**2).sum()/((yt - yt.mean(0))**2).sum()
    return beta, r2
def with_comp(E, beta, keep):
    F = {k: v.clone() for k,v in E.items()}; F['W2'][:, keep] += torch.tensor(beta[:-1].T)
    F['b2'] = torch.tensor(b2o + beta[-1]); return F
four = [26,41,131,159]; keep = [j for j in range(192) if j not in four]
for lab, E in [('ins_only(0)', ins_only(0.0)), ('gated3(0,0)', gated3(0.0,0.0)), ('ins_only(-2.7)', gi), ('gated3(-2.7,2)', gated3(-2.7,2.0))]:
    d, s = costm(E); print(f'{lab:>16} test-half all {d[te].mean():.4f}', s)
for lost in [[26,41],[26,41,131,159]]:
    beta, r2 = comp_fit(lost, keep); print('comp', lost, 'test R2 of lost contribution', round(r2,3))
    for lab, E in [('ins_only(-2.7)', gi), ('gated3(-2.7,2)', gated3(-2.7,2.0))]:
        if lost==[26,41] and lab.startswith('gated3'): continue
        d, s = costm(with_comp(E, beta, keep)); print(f'   +comp {lab:>14} test-half all {d[te].mean():.4f}', s)
```