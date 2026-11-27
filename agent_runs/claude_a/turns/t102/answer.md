No h dimension is truly unused: every LayerNorm gain is between 0.8 and 1.2. The least-read dimensions are small, though. Their spread across bundles is about 2.3–3.5, while a slot token's spread across dimensions is about 65. Each carries only a few percent of the normalised signal.

**LayerNorm gains vary moderately.** The σ used by these LayerNorms changes by about 11% across bundles, so effect sizes would vary by roughly that much.

**Rate decodability from the head-3 value of u:**

| Drug | R², head-3 value (48 dims) | R², full u (192 dims) |
|---|---|---|
| Insulin | 0.995 | 0.9995 |
| Norepinephrine | 0.94 | 0.994 |

**Revised plan (Plan B).**
- *Memory.* Each memory channel is an antisymmetric pair of low-visibility h dimensions (+m and −m). All constant terms cancel exactly through every LayerNorm.
  - Write: head 3 writes the rate.
  - Persistence: `flow.a` sets each channel's decay.
  - Isolation: the other writers to these dimensions are zeroed.
- *Readout.* A code-gated GELU unit in the value head reads the memory. It also subtracts a regression-based estimate of the expected dose effect, taken from the head's input, which cancels the mean dose effect the model already includes. Its weights would be fitted by weighted least squares on teacher-forced residuals.

**Before editing.** I need a full teacher-forced pipeline I control: sequential core steps plus the value head and its NLL.

**This run.** Check that the existing helpers reproduce the stored results:
- `ctx_fn` and `zpost_mean` should reproduce the stored z posterior.
- `head_value` should reproduce the stored 4-sample probabilities.

Then build a sequential teacher-forced run and confirm it matches the stored states and the tool's baseline NLL (1.40394). Matching numbers confirm the pipeline; mismatches tell me which helper's conventions I have misread.

```python
import time
t0_ = time.time()
# (a) ctx/zpost check on stored states
prevs = SB['prev'].values
ok_rows = np.where(~np.isnan(prevs))[0]
pr_ = prevs[ok_rows].astype(int)
dts_t = torch.tensor(SB['dts'].values[ok_rows], dtype=torch.float32)
try:
    cx = ctx_fn(OUT[pr_], HS[pr_], dts_t, PO)
    zp_ = zpost_mean(cx, eALL[ok_rows], uALL[ok_rows], PO)
    print('zpost check max|dz|', (zp_ - ZP[ok_rows]).abs().max().item(), 'z scale', ZP.abs().mean().item())
except Exception as ex:
    print('ctx/zpost error', repr(ex))
# (b) head_value check vs stored p4
try:
    ph_ = head_value(OUT[prevrow], HS[prevrow], dtv, cidx, hasl, ages, zl, lvbt, eps4, PO)
    if isinstance(ph_, tuple): print('head_value returns tuple of', [getattr(t,'shape',None) for t in ph_]); ph0 = ph_[0]
    else: ph0 = ph_
    print('head_value vs p4 max diff', (ph0 - p4).abs().max().item())
except Exception as ex:
    print('head_value error', repr(ex))
print('obs_b range', obs_b.min(), obs_b.max(), 'nll4 mean', nll4.mean(), ' check -log p4[obs]', (-torch.log(p4[torch.arange(len(obs_b)), torch.tensor(obs_b)])).mean().item())
print('prevrow sample', prevrow[:5], 'dtv', dtv[:5], 'codes', codes_[:5], 'kinds Rr cols', Rr.columns.tolist())
print(time.time() - t0_)
```