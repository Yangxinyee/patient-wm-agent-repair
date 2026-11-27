The decoding test is discouraging (I'll suppress the pandas warnings from now on). Linear decoders from with_code:

| Target | R² (recorded doses) | Arm-change slope |
|---|---|---|
| Lag-1 insulin dose | 0.63 | 0.4–0.85 (x1/2 best at 0.85, x2 0.57, x0 0.54) |
| Lag-1 policy dose | 0.95 | (does not change across arms here) |
| Lag-1 deviation | negative | similar attenuation |
| kap = 1 combined signal | 0.16 | 0.25–0.65 |

Decoding from the hidden activations was worse.

So at the head the dose is represented weakly and saturates for larger doses. A head-only tilt would add noise to factual predictions and would need about 1.8x amplification, which risks the 0.005-nat factual budget.

Before giving up on a head-only edit, I need to know where along the path the dose information is lost:
- u (the encoder output for the drug atoms),
- the prelude output,
- the pre-coda state hn,
- the new state h (all 4 slots, 768 numbers),
- out,
- with_code.

If an early stage (u, or the full 768-number state) decodes the lag-1 dose with R² near 1 and slope near 1, I can add a small relay. A repurposed hidden unit in the coda or prelude feed-forward layer would read the dose there and write it into a dedicated direction of `out`, which the head can then read cleanly. If even u is attenuated, the edit has to start at the encoder.

This run collects those stages at every insulin bundle, teacher-forced, on the same 300 stays and arms. It fits cross-validated ridge decoders for the bundle's insulin rate and its policy deviation.

```python
import warnings; warnings.filterwarnings('ignore')
A_=RD['A']; codeA=A_.code.values; kindA=A_.kind.values
insS=np.where((codeA==CODE['insulin'])&(kindA==0))[0]
multsD=[arm_mult(RD,a,'insulin') for a in armsD]
ins_by_sk={}
for j in insS:
    ins_by_sk[(A_.si.values[j], int(A_.k.values[j]) if 'k' in A_.columns else None)]=j
print('A cols',list(A_.columns)[:30])
```