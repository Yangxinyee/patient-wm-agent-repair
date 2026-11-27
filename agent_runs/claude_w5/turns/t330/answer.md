**What fwd_10 showed.** The saved version fails the factual constraint by a wide margin. It used c_I=1.5 and c_N=1.3, applied by rescaling b_in. On 100 tuning stays not used before, its value NLL is 1.66997 against 1.65072 for the original, which is +0.0193 nats against an allowed +0.005. Every reading type got worse:

| Reading | NLL change |
|---|---|
| glucose_lab | +0.033 |
| map_cuff | +0.024 |
| glucose_finger | +0.017 |
| map_arterial | +0.005 |

**What I conclude.**
- As it stands, the edit would be disqualified however good its dose readouts are.
- The damage is largest on readings the edit never targets (lab glucose, cuff MAP). An over-strong drug effect would mainly hurt the readings it acts on. So the likely cause is collateral damage: rescaling b_in along q_I and q_N amplifies everything in the model's state along those directions, not just my injected drug effect.
- An earlier factual check of the c=1 version was fine on training stays (−0.0032), but that check never involved the b_in rescaling.

**What I do now.**
1. Put b_in back to the c=1 version. This is the safe fallback state and is written first.
2. Apply the same gain increase only where my injected effect is produced. I recover the rescaling as an exact rank-2 matrix M from the two b_in versions. I apply (I+M) only to the edited output columns of prelude.0's feed-forward layer. The rest of the state is then no longer amplified.
3. Back up the c=1 version of that layer, check the structure (residual of the rank-2 fit, whether other layers sit between prelude.0 and b_in), and log everything.

**Next check.** A factual pass on the same tuning stays C, original and edited together.
- **Confirm:** the edited model's NLL is within about +0.002 of 1.65072, or better. Then a dose_edit pass on C tests whether insulin x2 still lands near −14.
- **Refute:** if it is still above about +0.005, the remaining runs go to the plain c=1 version (run 299), and to dropping all edits if even that fails (run 300).

```python
def _r298():
    import numpy as np, torch
    def _np(t):
        return t.detach().cpu().numpy().astype(np.float64) if hasattr(t, 'detach') else np.asarray(t, dtype=np.float64)
    as_t = hasattr(load('core.b_in.weight'), 'detach')
    def _back(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        return torch.from_numpy(a) if as_t else a
    nb, n2 = 'core.b_in.weight', 'core.prelude.0.ffn.2.weight'
    n0w, n0b = 'core.prelude.0.ffn.0.weight', 'core.prelude.0.ffn.0.bias'
    pl = sorted({n.split('.')[2] for n in NAMES if n.startswith('core.prelude.')})
    near = [n for n in NAMES if n.startswith('core.') and ('norm' in n or 'b_in' in n) and 'prelude.0.' not in n][:8]
    print("prelude layers:", pl, "| core norm/b_in names:", near)
    B1 = _np(globals()['B_IN_c1']); Bc = _np(load(nb)); Bo = _np(load_original(nb))
    D = Bc - B1
    M = np.linalg.solve(B1, D)
    U, S, Vt = np.linalg.svd(M)
    M2 = (U[:, :2] * S[:2]) @ Vt[:2]
    r2 = np.linalg.norm(B1 @ M2 - D) / np.linalg.norm(D)
    print("B1-orig maxdiff %.3g | M sv %s | rank2 resid %.2e" % (np.abs(B1 - Bo).max(), np.round(S[:3], 4), r2))
    if np.abs(B1 - Bo).max() == 0:
        drop_edit(nb)
    else:
        save_edit(nb, _back(B1))
    W2e = _np(load(n2)); W2o = _np(load_original(n2))
    np.save('/work/ffn2_c1.npy', W2e.astype(np.float32))
    ok = bool((W2e.shape[0] == M2.shape[0]) and r2 < 1e-3)
    if ok:
        W0e = _np(load(n0w)); W0o = _np(load_original(n0w)); b0e = _np(load(n0b)); b0o = _np(load_original(n0b))
        chg = np.where(np.abs(W2e - W2o).max(0) > 1e-7)[0]
        rep = set(np.where((np.abs(W0e - W0o).max(1) > 1e-7) | (np.abs(b0e - b0o) > 1e-7))[0].tolist())
        full = [int(j) for j in chg if j in rep]; outonly = [int(j) for j in chg if j not in rep]
        IM = np.eye(M2.shape[0]) + M2
        W2n = W2e.copy()
        if full:
            W2n[:, full] = IM @ W2e[:, full]
        if outonly:
            W2n[:, outonly] = W2o[:, outonly] + IM @ (W2e[:, outonly] - W2o[:, outonly])
        rel = np.linalg.norm(W2n - W2e) / max(np.linalg.norm(W2e[:, chg]), 1e-12)
        print("ffn2", W2e.shape, "chg cols", len(chg), "repurposed", len(full), "out-only", len(outonly),
              "rep w/o col chg", len(rep - set(chg.tolist())), "rel change %.4f" % rel)
        save_edit(n2, _back(W2n))
    else:
        print("ffn2 scaling skipped (shape/resid check failed)")
    print("edited now:", edited())
    with open('/work/notes.md', 'a') as f:
        f.write("- fwd_10 factual both tuning C, c_I=1.5/c_N=1.3 via b_in: orig 1.65072 edited 1.66997 (+0.0193) FAILS constraint "
                "(glc_lab +.033, map_cuff +.024, glc_f +.017, map_art +.005) -> b_in scaling damages broadly.\n"
                "- Run 298: b_in -> c=1 (B_IN_c1); kappa scaling emulated only on edited ffn.2 cols: W2[:,j]=(I+M)W2[:,j], "
                "M=solve(B1,Dbin) rank2 (ok=%s). Backup c=1 ffn2: /work/ffn2_c1.npy. Next: factual C, dose_edit C. "
                "Fallback: ffn2_c1 (pure c=1), then drop all.\n" % ok)
_r298()
```