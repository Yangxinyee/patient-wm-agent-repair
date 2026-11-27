Adding the bin-table amplification changed almost nothing. Insulin x2 went to −2.54 (vs −2.51 with the z columns alone) and x1/2 to +1.69 (vs +1.66); norepinephrine stayed near zero.

So input-level amplification is exhausted. Either the dose-varying part of the bin tables along this profile is tiny, or the encoder saturates. In both cases the dose signal must be amplified after the action encoder. The dose signal can reach the state by two routes: the action token u_k directly in the prelude, or the z posterior (dimension 26) via `z_in`. Each candidate edit there carries side effects:
- `z_in` along z26 would also amplify z26 at reading bundles.
- Scaling z_post's u-block would also amplify the constant "a drug was given" part.

To choose, I first run a mediation check on the original model's states, with no forward pass. At insulin bundles I regress the state change on z26 together with the dose innovation and controls. If adding z26 removes most of the innovation coefficient, the dose effect flows mainly through z, and the z route is the lever. If the innovation coefficient barely changes, the action token in the prelude carries it.

```python
def mediation(Sx, label):
    x = Sx['x']; sidx = Sx['stay_id']
    same = np.r_[False, sidx[1:]==sidx[:-1]]
    sel = same[ins_b] & ok
    rr = ins_b[sel]
    D_ = x[rr] - x[rr-1]
    z26_ = x[rr, 960+26]
    base_cols = [np.ones(sel.sum()), innov[sel], exp_rate[sel], (gsrc_b[sel]-200)/50, hn3[rr].astype(float)]
    # controls: previous-state summary (top PCs of previous out)
    po = x[rr-1, 0:192]; po = po - po.mean(0); U_, s_, Vt_ = np.linalg.svd(po, full_matrices=False); pcs = U_[:, :10]*s_[:10]
    A1 = np.column_stack(base_cols + [pcs])
    A2 = np.column_stack(base_cols + [pcs, z26_])
    for nm_, sl in [('out', slice(0,192)), ('h', slice(192,960))]:
        c1, *_ = np.linalg.lstsq(A1, D_[:, sl], rcond=None)
        c2, *_ = np.linalg.lstsq(A2, D_[:, sl], rcond=None)
        b1 = c1[1]; b2 = c2[1]; bz = c2[-1]
        print(label, nm_, "|innov coef| w/o z26 %.3f, with z26 %.3f; cos %.2f; |z26 coef| %.3f; z26-mediated part %.3f" %
              (np.linalg.norm(b1), np.linalg.norm(b2), b1@b2/np.linalg.norm(b1)/np.linalg.norm(b2), np.linalg.norm(bz), np.linalg.norm(bz)*0.66))
mediation(So, 'orig'); mediation(Se, 'edit')
```