Freezing a few hidden units is essentially free. Units 35, 26, 24 and 125 (and 180 as a spare) have activation sd of 0.001–0.006, and the second-order cost of fixing each at its mean is below 1e-6 nats per reading. So I can build the two gated pairs at no measurable factual cost:
- Pair 35/26 for glucose: one unit reads the dose probe, the other is the gate-only reference, so the gate's variation cancels.
- Pair 24/125 for MAP.

My numpy rebuild of the value head is close but not exact (mean MAE about 2 mg/dL against the tool). That is good enough for costs and probes, not for final numbers.

The correction will tilt the logits along the bin centres. For roughly Gaussian predictions this is a translation of size β·σ².

What the probe should read is the critical point. A ridge probe of the dose shock is shrunk, responding only about R² ≈ 0.4 per unit of real dose change. So a gain set for the best factual fit will under-respond in rollouts. I will calibrate the gain with dose_edit and check what the extra gain costs factually.

To fit the probes I need with_code vectors aligned with the tool's factual errors and with the dose shocks at each lag. I have this for 99 stays (5521 readings), and apparently for another 300 stays (21904 readings, from forward passes 5–7, which my notebook tail does not describe). This run checks what those passes contain and how the arrays line up, so I don't repeat or misuse a forward pass.

```python
import os, json
for k_ in [5,6,7]:
    dd = f'/work/fwd_{k_}'
    print(k_, sorted(os.listdir(dd)))
for nm_ in ['D5','D99','FF','GF','G','P','PT']:
    vv = globals().get(nm_)
    if isinstance(vv, dict):
        print(nm_, {kk_: (getattr(x__,'shape',None) if hasattr(x__,'shape') else (len(x__) if hasattr(x__,'__len__') else x__)) for kk_, x__ in vv.items()})
print("S5 files:", S5.files, [S5[f].shape for f in S5.files])
n3 = json.load(open('/work/new300.json')); print("new300", type(n3).__name__, len(n3), n3[:3] if isinstance(n3, list) else list(n3)[:3])
ss_ = json.load(open('/work/sel_states.json')); print("sel_states", len(ss_), ss_[:3])
print("eps_ins/eps_ne", eps_ins.shape, eps_ne.shape, "ids_arr", ids_arr[:3], "sidx type", type(sidx).__name__)
print("WCall rows 21904; stays covered by codeall? check S5/S6/S7 stay counts:",
      [len(np.unique(S['stay'])) if 'stay' in S.files else None for S in (S5,S6,S7)])
print(open('/work/notes.md').read()[-2500:])
```