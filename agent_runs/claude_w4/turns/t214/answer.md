Restricting δ to head 1's 48-dimensional output space is fatal. The required norms rise to 57 per mg/dL for glucose and 165 per mmHg for MAP, and the FFN then responds nonlinearly, with NE leaking into glucose. Without the restriction δ_I needs only 13.2 and passes through the FFN linearly.

So I'll add one rank-1 term to the output projection (attn.out) as well. Head 1's value space gets three nearly unused dimensions, chosen for minimal original activity:
- one carries the exact insulin rate, mapped to −g_I·δ_I;
- one carries NE rate minus a constant expected rate, mapped to +g_N·δ_N (no FFN needed for NE);
- one carries insulin presence, mapped to a carrier direction whose normalised natural projection is about zero, used to gate the FFN correction.

This step chooses those directions and constructs the edited value and output weights. A small original activity along the chosen value dimensions means negligible leakage.

```python
V1o = load_original('core.prelude.0.attn.kv.weight').numpy()[240:288]
attd_all = r2['att'][:,1,:4,:].numpy(); attn_all = r2n['att'][:,1,:4,:].numpy()
vd = np.einsum('njt,ntd->njd', attd_all, L1T @ V1o.T).reshape(-1,48); vn = np.einsum('njt,ntd->njd', attn_all, L1Tn @ V1o.T).reshape(-1,48)
Sv = np.r_[vd, vn].T @ np.r_[vd, vn] / (len(vd)+len(vn))
evs, evv = np.linalg.eigh(Sv)
print('head-1 value 2nd-moment smallest', np.round(evs[:4],5), 'largest %.1f' % evs[-1])
aI3, aN3, aP3 = evv[:,0], evv[:,1], evv[:,2]
# normalized-token second moment for presence carrier direction q
n0 = (x1all - x1all.mean(1,keepdims=True)) / x1all.std(1,keepdims=True)
M0 = n0.T @ n0 / len(n0)
e0, v0 = np.linalg.eigh(M0)
qd = v0[:,1] - v0[:,1].mean(); qd /= np.linalg.norm(qd)
print('M0 smallest', np.round(e0[:3],8), ' q nat. normalized proj rms %.2e' % np.sqrt(((n0@qd)**2).mean()))
c0N = float(rN[iN].mean()); print('c0 NE (mean rate when on, sel100) %.3f' % c0N)
TT['fN'] = TT['tN'] - c0N*TT['pN']
BB3 = {}
for nm in ['tI','fN','pI']:
    if nm=='tI': BB3[nm] = BB['tI_p']
    else:
        BB3[nm] = np.linalg.solve(G_ + 1e-6*np.eye(192), A_.T@(TT[nm]*sw))
    e_ = FF@BB3[nm] - TT[nm]
    print(nm, '|b| %.0f rms %.4f max %.3f' % (np.linalg.norm(BB3[nm]), np.sqrt((e_**2).mean()), np.abs(e_).max()))
def build_v3(gI, gN, sP=2.0):
    kv_ = load_original('core.prelude.0.attn.kv.weight').clone().numpy()
    wo_ = load_original('core.prelude.0.attn.out.weight').clone().numpy()
    W1o = wo_[:, 48:96].copy()
    kv_[240:288] += np.outer(aI3, BB3['tI']) + np.outer(aN3, BB3['fN']) + np.outer(aP3, BB3['pI'])
    for a_, tgt in [(aI3, -gI*dI), (aN3, gN*dN), (aP3, sP*qd)]:
        wo_[:, 48:96] += np.outer(tgt - W1o@a_, a_)
    return torch.tensor(kv_, dtype=torch.float32), torch.tensor(wo_, dtype=torch.float32)
print('built OK')
```