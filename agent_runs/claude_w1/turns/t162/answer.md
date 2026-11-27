Head 2 is the clean channel. Its slot queries attend 100% to the u token on both drug and reading bundles, and on reading bundles u is the identical empty-set vector. Heads 0, 1 and 3 attend partly to the measurement token. A value edit confined to head 2, with a decoder that is exactly zero on the empty-set token, therefore injects only at drug bundles, in proportion to the decoded rate.

This run fits ridge decoders for the insulin and NE rates on norm1(u), with a hard zero on the empty token, and checks their accuracy. It also measures the model's own dose directions:
- the head-2 value slope per unit rate;
- the total attention-output slope across heads;
- how well head 2 alone can reproduce that total slope.

Decoder errors that are tiny relative to the innovation sd (0.37 u/h for insulin, 0.08 for NE) would confirm the building block.

```python
Wkv = Wo['core.prelude.0.attn.kv.weight']; Wout = Wo['core.prelude.0.attn.out.weight']
xe = lnp(ue[None], n1w, n1b)[0]; D = drug; Xd = Xu[D]
def dec(y, lam=1e-4, big=1e5):
    A = Xd.T@Xd + lam*np.eye(192) + big*np.outer(xe,xe); w = np.linalg.solve(A, Xd.T@y); return w, Xd@w
for nm, y in [('ins', DI[D]), ('ne', DN[D])]:
    w, p = dec(y); e = p-y; on = y>0
    print(nm, 'R2 %.5f errsd on %.4f off %.4f maxabs %.3f empty %.1e' % (1-e.var()/y.var(), e[on].std(), e[~on].std(), np.abs(e).max(), xe@w))
    globals()['w_'+nm] = w
Z = np.column_stack([np.ones(D.sum()), DI[D], DN[D]])
V2 = Xd@Wkv[288:336].T; c2, *_ = np.linalg.lstsq(Z, V2, rcond=None); r2v = 1-((V2-Z@c2)**2).sum()/((V2-V2.mean(0))**2).sum()
O = (Xd@Wkv[192:].T)@Wout.T; cO, *_ = np.linalg.lstsq(Z, O, rcond=None)
sI2, sN2 = c2[1], c2[2]; rI, rN = cO[1], cO[2]
h2I = Wout[:,96:144]@sI2; h2N = Wout[:,96:144]@sN2
cos = lambda a,b: a@b/np.linalg.norm(a)/np.linalg.norm(b)
print('head2 V linear-fit R2 %.3f |sI2| %.3f |sN2| %.3f' % (r2v, np.linalg.norm(sI2), np.linalg.norm(sN2)))
print('total attn-out slope |rI| %.3f |rN| %.3f | head2 part |h2I| %.3f cos %.3f |h2N| %.3f cos %.3f' % (np.linalg.norm(rI), np.linalg.norm(rN), np.linalg.norm(h2I), cos(h2I,rI), np.linalg.norm(h2N), cos(h2N,rN)))
Pj = Wout[:,96:144]; best = Pj@np.linalg.lstsq(Pj, rI, rcond=None)[0]; print('best head2 approx of rI cos %.3f' % cos(best, rI))
```