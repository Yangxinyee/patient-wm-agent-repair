The tensor list settles the design question:
- The glucose value head is a small two-layer MLP (192→192→17) on `with_code`, plus a per-(code, last-bin) logit table, so it can be copied exactly on CPU.
- The state is rebuilt each step by two recurrent layers with memory cross-attention, then c_out and a transformer-layer coda. Dose information can only enter the state through the prelude's u-token attention and its FFN.
- Of the networks that see both the state summary and u_k side by side, only z_post (where v1 lived) does. Its z output is ignored by the prelude's attention, which is why v1 had no effect.

That leaves the prelude FFN as the realistic place to compute the dose surprise, s = (rate read from u) − E[rate | state], at drug bundles. Its input for each slot token is LN(h_slot + attention outputs), and heads 1 to 3 copy u_k into it at drug bundles.

Next run:
- rewrite the prelude so it returns its intermediate values, and check it against my earlier copy (E0);
- test, with ridge and linear-SVM fits on the FFN input of each slot, whether three things are linearly readable: the insulin rate, the insulin-present flag (with measurement bundles as negatives) and the NE rate.

What would decide it: a high cross-validated R² for the rate and clean separation for the flag would mean gated FFN units can compute the surprise, and v2 goes ahead in the FFN. Poor readouts would mean I first need a cleaner u channel, by editing head 1's value path.

```python
import torch, numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.svm import LinearSVC
Wp = {n: load(n) for n in NAMES if n.startswith('core.prelude')}
def LNx(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdim=True); v = x.var(-1, unbiased=False, keepdim=True)
    return (x - m) / torch.sqrt(v + eps) * w + b
def prelude2(TOK, W):
    p = 'core.prelude.0.'
    B, T, _ = TOK.shape
    n1 = LNx(TOK, W[p+'norm1.weight'], W[p+'norm1.bias'])
    q = n1 @ W[p+'attn.q.weight'].T
    kv = n1 @ W[p+'attn.kv.weight'].T
    sp = lambda a: a.reshape(B, T, 4, 48).transpose(1, 2)
    att = torch.softmax(sp(q) @ sp(kv[..., :192]).transpose(-1, -2) / 48**0.5, -1)
    ao = (att @ sp(kv[..., 192:])).transpose(1, 2).reshape(B, T, 192)
    x1 = TOK + ao @ W[p+'attn.out.weight'].T
    n3 = LNx(x1, W[p+'norm3.weight'], W[p+'norm3.bias'])
    pre = n3 @ W[p+'ffn.0.weight'].T + W[p+'ffn.0.bias']
    hid = torch.nn.functional.gelu(pre)
    x2 = x1 + hid @ W[p+'ffn.2.weight'].T + W[p+'ffn.2.bias']
    E = LNx(x2[:, :4], W['core.prelude_norm.weight'], W['core.prelude_norm.bias'])
    return dict(E=E, x1=x1, n3=n3, pre=pre, hid=hid, x2=x2, att=att)
Pd = prelude2(TOK.float(), Wp)
print('E check vs E0: maxabs', float((Pd['E'] - E0).abs().max()))
Pn = prelude2(TOKn.float(), Wp)
rI = np.asarray(RI_d, float); rN = np.asarray(RN_d, float)
print('drug bundles', len(rI), 'ins>0', int((rI > 0).sum()), 'ne>0', int((rN > 0).sum()), 'meas bundles', TOKn.shape[0])
grpd = np.asarray(sidd)
for i in range(4):
    X = Pd['n3'][:, i].numpy()
    mi = rI > 0
    pr = cross_val_predict(Ridge(alpha=1.0), X[mi], rI[mi], groups=grpd[mi], cv=GroupKFold(5))
    r2I = 1 - ((pr - rI[mi])**2).mean() / rI[mi].var()
    mn = rN > 0
    prn = cross_val_predict(Ridge(alpha=1.0), X[mn], rN[mn], groups=grpd[mn], cv=GroupKFold(5))
    r2N = 1 - ((prn - rN[mn])**2).mean() / rN[mn].var()
    Xg = np.vstack([X, Pn['n3'][:, i].numpy()]); yg = np.r_[(rI > 0).astype(int), np.zeros(TOKn.shape[0], int)]
    clf_ = LinearSVC(C=10, max_iter=20000).fit(Xg, yg); sc_ = clf_.decision_function(Xg)
    print(f'slot{i}: R2 insulin rate {r2I:.3f}  NE rate {r2N:.3f}  | gate ins: min pos {sc_[yg==1].min():.2f} max neg {sc_[yg==0].max():.2f} '
          f'std n3 input x1: {float(Pd["x1"][:, i].std(-1).mean()):.2f}±{float(Pd["x1"][:, i].std(-1).std()):.2f}')
print('tu cols', list(tu.columns)); print(tu.head(3).to_string())
```