The context passes this check. Every reasonable context variant gives the same prior correlation of about 0.50 with the saved posterior. The posterior gets substantial information from the bundle's own encoding, and my set-encoder output does not reproduce it. Among my earlier variants, putting the encoding in the third input block ([ctx, u, e]) did best (0.29).

Two assumptions remain untested: which activation the MLPs use (the architecture file mentions relu somewhere, but not necessarily everywhere), and where the size term enters the set encoder. This run prints the relu context from the architecture file. It then tests GELU versus ReLU in the encoder and in `z_post`, crossed with the size-term placements, under both block orderings.

If a variant pushes the posterior correlation clearly above 0.5, the encoders become reproducible and a numpy rebuild of the core is back on. If not, I will stop and switch to a black-box strategy driven by forward passes.

```python
import re
arch_lines = [l_ for l_ in open('/model/ARCHITECTURE.md').read().splitlines() if re.search('relu|gelu|act', l_, re.I)]
print(arch_lines)
from scipy.special import erf
gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2)))
def mlp_a(x, p, act): return lin(act(lin(x, p + '.0')), p + '.2')
def enc_a(vecs, p, act, size_mode):
    V_ = np.array(vecs); M1 = mlp_a(V_, p + '.mlp1', act); n_ = len(V_); s_ = M1.sum(0) / np.sqrt(n_)
    sz = lin(np.array([np.log1p(n_)]), p + '.size')
    if size_mode == 'in': y_ = mlp_a(s_ + sz, p + '.mlp2', act)
    elif size_mode == 'out': y_ = mlp_a(s_, p + '.mlp2', act) + sz
    else: y_ = mlp_a(s_, p + '.mlp2', act)
    return ln(y_, W[p + '.norm.weight'], W[p + '.norm.bias'])
res_ = []
for enc_act_n, enc_act in [('relu', relu), ('gelu', gelu)]:
    for z_act_n, z_act in [('relu', relu), ('gelu', gelu)]:
        for size_mode in ['in', 'out', 'none']:
            for order in ['ceu', 'cue']:
                P_ = []
                for c_, vecs, t_ in pre_c:
                    e_k = enc_a(vecs, 'encode', enc_act, size_mode); parts = {'c': c_, 'e': e_k, 'u': np.zeros(192)}
                    P_.append(mlp_a(np.concatenate([parts[k_] for k_ in order]), 'z_post', z_act)[:32])
                P_ = np.array(P_)
                res_.append((np.mean([np.corrcoef(P_[:, j], T_all[:, j])[0, 1] for j in range(32)]), enc_act_n, z_act_n, size_mode, order))
res_.sort(reverse=True)
for r_ in res_[:8]: print(r_)
```