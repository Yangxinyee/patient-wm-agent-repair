The constraint holds comfortably. The edited model's factual value NLL is 1.43319 against 1.43643 for the original, an improvement of 0.0032 nats, and every event type improves. So there is room to strengthen the edit.

Current readouts on selection A, under the assumption that each readout scales roughly linearly with the injection gains (original + c × (edited − original)):

| Drug | Gain c that puts x2 on target | x1/2 at that gain | Gain c minimising squared error |
|---|---|---|---|
| insulin | 1.55 | ≈ −2.5 | 1.35 |
| norepinephrine | 1.30 | ≈ −0.46 | 1.34 |

The insulin x1/2 readout cannot be fixed by gain alone. Its even component (about −6.7 at hour 24) looks like the mirror image of the factual-path drift (+4.6). That suggests the rollout encodes the factual-path drug atoms in a way that biases the decoder, and any rescaling removes that bias. I can't pin it down with the budget left.

So I'll scale the gains: c_I = 1.5, c_N = 1.3. The gain sits in b_in's columns along the quiet directions q_I and q_N, so I'll rescale W·q_k, after checking that q_I and q_N are orthonormal. Then I'll log the full story in the notebook.

```python
import numpy as np, torch
qI, qN = np.asarray(EP_['q_I'], float), np.asarray(EP_['q_N'], float)
print('|qI| %.6f |qN| %.6f qI.qN %.2e' % (np.linalg.norm(qI), np.linalg.norm(qN), qI@qN))
Wb = load('core.b_in.weight').double().numpy(); Wb0 = load_original('core.b_in.weight').double().numpy()
Q = np.stack([qI, qN], 1); Qd = Q @ np.linalg.inv(Q.T @ Q)           # dual basis (exact even if not orthonormal)
colI, colN = Wb @ qI, Wb @ qN
print('|W qI| %.2f |W qN| %.2f (orig %.3f %.3f)' % (np.linalg.norm(colI), np.linalg.norm(colN), np.linalg.norm(Wb0@qI), np.linalg.norm(Wb0@qN)))
cI_, cN_ = 1.5, 1.3
Wnew = Wb + (cI_-1)*np.outer(colI, Qd[:,0]) + (cN_-1)*np.outer(colN, Qd[:,1])
print('check: Wnew qI / W qI = %.4f, Wnew qN / W qN = %.4f' % ((Wnew@qI)@colI/(colI@colI), (Wnew@qN)@colN/(colN@colN)))
B_IN_c1 = Wb.copy()          # keep the c=1 version
save_edit('core.b_in.weight', torch.tensor(Wnew, dtype=torch.float32)); WE['core.b_in.weight'] = torch.tensor(Wnew, dtype=torch.float64)
print('edited:', edited())
with open('/work/notes.md','a') as f: f.write("""
## Runs 262-283 (restart 6)
- fwd_6 (dose_edit both selA, FULL edit incl kv, hash ee4374ba4c): ins x2 -9.43, x1/2 -1.63, ne x2 +3.21, x1/2 -0.40 (orig -0.55, 0.00, 0.19, -0.17).
  Per hour ins x2 -3.1 (h13) -> -14.1 (h24); factual path drift edited-orig: glucose +0.7 -> +4.6, MAP -1.4 -> -6.8.
- fwd_7 states both (13 tuning stays): tool edited == local WE (rel err 2e-6). W64 had the kv edit too (in-place bug) -> FIXED (W64 = true original now).
- Local single-bundle responses linear & symmetric (slopes ~0.9 x2/x1/2). Local h13 (teacher forced to 12:05): ins x2 -2.66, x1/2 +1.28; tool -3.08/+0.09.
  Tool h13 factual of EDITED model differs from local by median 2.1 mg/dL (orig 0.37) -> dose_edit rollout encodes something differently
  (not u=0, not z sampling, not END/dose drop; right-side bins for h12 atoms reproduce tool x2 at h13 exactly (-3.10) but not x1/2). UNRESOLVED.
- Without stage 1 (kv), (rate-c_hat) poorly decodable from n3 (R2 .72 I / .38 N) -> keep stage 1.
- fwd_8 factual both train selB: orig 1.43643, edited 1.43319 (-0.0032) -> constraint OK.
- Run 283: kappa scaling via b_in columns along q_I,q_N: c_I=1.5, c_N=1.3 (B_IN_c1 = c=1 version kept in kernel). Next: dose_edit selA + factual check.
""")
```