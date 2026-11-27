`fN` is not broken; it is centred. At NE bundles it equals the NE rate minus 0.52 (slope 0.99, residual rms 0.03–0.05 for both NE-only and two-drug bundles), and it is about 0 when NE is absent. So the NE carrier already subtracts a constant "typical rate", while the insulin carrier `tI` is uncentred. An uncentred NE readout, `BB['tN']`, also exists (rms 0.05).

For the dose questions, centring doesn't matter to first order. Under ×2 the readout changes by exactly the recorded rate either way. So I can calibrate the carrier gain now and deal with the factual-fit subtraction afterwards.

Next I'll save v3 with gI = 1, gN = 1, sP = 0. That means insulin pushes the slots along −dI per unit/h and NE along +dN per mcg/kg/min above 0.52. Then I'll run a dose_edit on the same 100 tuning stays (sel100) with "both", so the original and edited models share random numbers.

The estimated truths on these stays are about −13 / +6.5 mg/dL for insulin ×2 / ×½ and +4.9 / −2.5 mmHg for NE ×2 / ×½. The original model gives −0.35 / +0.21 / +0.02 / −0.25.
- **Carrier works:** the edited insulin ×2 shift moves clearly beyond −0.35 and NE ×2 clearly above +0.02, with ×½ roughly half and opposite. That gives the gain per unit for calibration.
- **Path is ineffective:** no change, as happened with v1 through the z token.

```python
kv_n, wo_n = build_v3(1.0, 1.0, 0.0)
kv_n=torch.as_tensor(np.asarray(kv_n),dtype=torch.float32); wo_n=torch.as_tensor(np.asarray(wo_n),dtype=torch.float32)
save_edit('core.prelude.0.attn.kv.weight', kv_n); save_edit('core.prelude.0.attn.out.weight', wo_n)
print('edited:', edited())
d1=(load('core.prelude.0.attn.kv.weight')-load_original('core.prelude.0.attn.kv.weight')).abs().sum(1).nonzero().ravel()
d2=(load('core.prelude.0.attn.out.weight')-load_original('core.prelude.0.attn.out.weight')).abs().sum(0).nonzero().ravel()
print('kv rows', d1.min().item(), d1.max().item(), 'wo cols', d2.min().item(), d2.max().item())
np.savez('/work/v3_g1_1_0.npz', kv=kv_n.numpy(), wo=wo_n.numpy())
import json; print('sel100 file len', len(json.load(open('/work/sel100_both.json'))))
with open('/work/notes.md','a') as fh: fh.write('''
## Restart 7 (runs 198-204)
- build_v3(gI,gN,sP) decoded: kv rows 240:288 += outer(aI3,BB3.tI)+outer(aN3,BB3.fN)+outer(aP3,BB3.pI); Wo[:,48:96] remapped so
  aI3 -> -gI*dI, aN3 -> +gN*dN, aP3 -> +sP*qd (a-dirs orthonormal; original value proj on them ~0, sd .001 at drug bundles).
- Readouts on LN1(u) at drug bundles: tI = insulin rate (rms .11); fN = (NE rate - 0.52)*present (rms .05), ~0.03 when absent;
  BB['tN'] = uncentred NE rate (rms .05); pI = insulin presence (sd .005). Slot norm ~795, |dI| 13.2, |dN| 28.7.
- Stored Jm/nuG arrays: semantics unclear (Jm@dI only cos .17 with -nuG) -> rely on forward passes for calibration.
- fwd_6 plan: dose_edit both, sel100 tuning, v3(gI=1,gN=1,sP=0). Truth sel100: ins x2 -12.96, x1/2 +6.48; NE x2 +4.94, x1/2 -2.47.
''')
```