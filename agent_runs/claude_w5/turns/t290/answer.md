Hour 13, which comes almost directly from the first edited bundle, is already asymmetric in the tool:

| Edit | Hour-13 shift |
|---|---|
| insulin x2 | −3.08 |
| insulin x1/2 | +0.09 (≈ +1.5 expected for a symmetric response) |
| norepinephrine x2 | +1.01 |
| norepinephrine x1/2 | −0.36 |

The edited model's factual rollout also drifts: glucose rises steadily to +4.6 and MAP falls to −6.8 by hour 24. That means the residual decoded inside the tool carries a negative bias, which my exact local model does not show. So the tool's single-step computation still differs from mine.

My main suspect is float32 precision. The kv edit has weights up to about 1900. The value projection can therefore produce entries in the thousands that must cancel, while the dose signal I need is a tiny variation on top. With float32 summation the rounding error could be as large as the signal.

To test this I'll measure, on real drug and reading bundles:
- how large the edited head-1 value vectors are, compared with how much they vary with the dose;
- what float32 rounding of the value product does to the decoded x.

I'll also get the 13 fwd_5 stay IDs for a possible states check of local against tool.

```python
import numpy as np
z5 = np.load('/work/fwd_5/states_original.npz'); ids5 = list(dict.fromkeys(int(s) for s in z5['stay_id']))
print('fwd5 stays', ids5)
print([n for n in NAMES if 'prelude' in n])
Wkv64 = to64(WE[kvn]); Wkv32 = Wkv64.astype(np.float32)
Wo_name = [n for n in NAMES if 'prelude.0.attn' in n and n != kvn and 'weight' in n]
print('attn weights', Wo_name, [SHAPES[n] for n in Wo_name])
# magnitudes at some drug & reading bundles
sid_ = SIDS[1]; Bs, Ss = bundles_B[sid_], steps_all[sid_]
for j in range(min(12, len(Bs))):
    R = internals(WE, Ss[j]); n1 = np.asarray(R['n1'], float)
    v64 = n1 @ Wkv64[240:288].T; v32 = (n1.astype(np.float32) @ Wkv32[240:288].T).astype(np.float64)
    vo = n1 @ to64(o_kv)[240:288].T
    a1 = np.asarray(R['att'], float)[1]   # head 1 attention (7x7)
    h1err = a1[:4] @ (v32 - v64)            # error in head-1 output at slots
    print(f'j{j} codes {[a["code"][:6] for a in Bs[j][1]]} |v_edit-v_orig| per token {np.round(np.abs(v64-vo).max(1),2)} | f32 err head1 out max {np.abs(h1err).max():.2e} | att slot->u {np.round(a1[:4,5].mean(),4)}')
```