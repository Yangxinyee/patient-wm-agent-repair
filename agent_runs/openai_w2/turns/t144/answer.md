The helper writes into `S["x"]`, but `S` is currently an NPZ file object: each lookup returns a fresh array, so those writes never reach the head calculation. That explains the zero derivatives without refuting the pathway. I’ll use a persistent array dictionary and repeat the calibration; it should recover the earlier responses.

```python
# Keep head-state updates persistent during each local calculation.
Slocal128 = {k: np.array(S[k], copy=True) for k in S.keys()}

def eval_hpre128(hp, return_states=False):
    global S
    saved128 = S
    try:
        S = Slocal128
        return read_hpre120(hp, return_states=return_states)
    finally:
        S = saved128

eps128 = 0.25
J128 = np.empty_like(J120)
base_mean128 = np.array(eval_hpre128(hpre119), copy=True)
for d128 in range(192):
    pp128 = torch.zeros_like(hpre119)
    pp128[:, :, d128] = eps128 * delta119[d128]
    a128 = np.array(eval_hpre128(hpre119 + pp128), copy=True)
    b128 = np.array(eval_hpre128(hpre119 - pp128), copy=True)
    J128[:, d128] = (a128-b128)/(2*eps128)

events128 = probe_head110["event"].to_numpy()
A128 = np.stack([J128[events128 == e].mean(axis=0)
                for e in ("glucose_finger", "map_arterial")])
V128 = A128.T @ np.linalg.pinv(A128 @ A128.T)
print("Derivative norm:", np.linalg.norm(J128))
print("Relative difference from J120:",
      np.linalg.norm(J128-J120)/np.linalg.norm(J120))
print("Direction norms:", np.linalg.norm(V128, axis=0))
print("Linear cross-response:", A128 @ V128)
for j128 in range(2):
    pp128 = (torch.tensor(V128[:, j128], dtype=torch.float32)
             * delta119)[None, None, :]
    shift128 = np.array(eval_hpre128(hpre119 + pp128), copy=True)-base_mean128
    print("Unit steering", j128,
          {e: float(shift128[events128 == e].mean())
           for e in ("glucose_finger", "map_arterial")})

fit128 = prelude_fits112["E"][0.3]
Wdose128 = fit128["B"][1:] / fit128["sd"][:, None]
bdose128 = fit128["B"][0] - fit128["mu"] @ Wdose128
gamma128 = np.asarray(load_original("core.prelude_norm.weight"))
beta128 = np.asarray(load_original("core.prelude_norm.bias"))
cv128 = 1.0/gamma128
cc128 = beta128 @ cv128
Edecoder128 = Wdose128 + np.outer(cv128/cc128, bdose128)
qf128 = prelude_E111 @ Edecoder128
print("Affine conversion error:",
      np.max(np.abs(qf128-(prelude_E111 @ Wdose128+bdose128))))
print("Factual residual RMS:", np.sqrt(np.mean(qf128**2, axis=(0,1))))

gain_raw128 = {}
for (drug128, factor128), ecf128 in prelude_cf_E111.items():
    j128 = ("insulin", "norepinephrine").index(drug128)
    dx128 = dose_actual_repair[:, j128]*(factor128-1)
    dq128 = ((ecf128-prelude_E111) @ Edecoder128)[:, :, j128].mean(axis=1)
    use128 = dx128 != 0
    gain128 = float(dx128[use128] @ dq128[use128] /
                    (dx128[use128] @ dx128[use128]))
    gain_raw128[(drug128, factor128)] = gain128
    print("Residual/raw-rate slope:", drug128, factor128, gain128)

sample_map128 = np.searchsorted(posterior_rows_repair, core_rows104)
assert np.array_equal(posterior_rows_repair[sample_map128], core_rows104)
print("Sample dose range:", dose_actual_repair[sample_map128].min(axis=0),
      dose_actual_repair[sample_map128].max(axis=0))
print("Benchmark stay IDs:", benchmark_stays)
print("Edits:", edited())
```
