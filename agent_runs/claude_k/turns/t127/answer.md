Found it. With e taken as the set encoder of an empty set (not zero), input order [ctx, e, u], and bin index counted with strict "<" against the edges, my reconstruction reproduces the tool's posterior z to 0.0006. That is float precision. So ctx, the action-atom embedding and the action set encoder are now exact for single new-record bundles.

Next I'll extend this to every action bundle. That means the end-of-infusion atoms (kind 2), which may or may not carry the ending record's rate and dose, the action-B atoms, and multi-atom bundles. I'll check the atom count against `n_atoms` and report the z error per variant. A variant at about 0.001 over all 2,491 action bundles gives me an exact u_k for any edited rate. With that I can build and verify a dose channel offline.

```python
def drug_atoms(i, t, end_mode, mult=(1.0, 1.0)):
    V = []
    for c, nm, Pn, m in [(8, 'insulin', ins, mult[0]), (9, 'norepinephrine', ne, mult[1])]:
        r_now = Pn[i, t + 2] if 0 <= t + 2 < Pn.shape[1] else 0; r_prev = Pn[i, t + 1] if t + 1 >= 0 else 0
        r_now = 0 if np.isnan(r_now) else r_now * m; r_prev = 0 if np.isnan(r_prev) else r_prev * m
        dfac = NUMV[(nm, 'dose')]['median'] / NUMV[(nm, 'rate')]['median']
        if r_now > 0:
            V.append(atom_vec(c, kind=0, ekind=2, zd=zsc_(nm, 'dose', r_now * dfac), hd=1.0, zr=zsc_(nm, 'rate', r_now), hr=1.0,
                              dbin=fb_(nm, 'dose', r_now * dfac), rbin=fb_(nm, 'rate', r_now)))
        if r_prev > 0:
            if end_mode == 'attrs':
                V.append(atom_vec(c, kind=2, ekind=2, zd=zsc_(nm, 'dose', r_prev * dfac), hd=1.0, zr=zsc_(nm, 'rate', r_prev), hr=1.0,
                                  dbin=fb_(nm, 'dose', r_prev * dfac), rbin=fb_(nm, 'rate', r_prev)))
            else:
                V.append(atom_vec(c, kind=2, ekind=2))
    return V
E_EMPTY = e_empty()
def z_action_err(end_mode, js):
    errs, nmis = [], 0
    for j in js:
        t = int(np.floor(hrS[j])); i = sid2i[int(sidS[j])]; V = drug_atoms(i, t, end_mode)
        if len(V) != S2['n_atoms'][j]: nmis += 1; continue
        ctx = ctx_from(xs[j - 1], (hrS[j] - hrS[j - 1]) * 3600)
        errs.append(np.abs(xs[j, 960:992] - zmean(ctx, E_EMPTY, setenc(np.stack(V), 'encode_action'))).mean())
    return np.mean(errs), np.max(errs), nmis
jsA = [j for j in np.where(isA)[0] if j > 0 and sidS[j - 1] == sidS[j]][:400]
for em in ('attrs', 'bare'):
    m, mx, nm_ = z_action_err(em, jsA); print(em, 'mean err %.4f max %.4f n-mismatch %d of %d' % (m, mx, nm_, len(jsA)))
```