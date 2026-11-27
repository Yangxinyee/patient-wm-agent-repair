Now I have the exact tensor layout, which changes the design:

- **State update constants:** `core.delta_in` and `core.a_in` are 192-vectors, not matrices. So in `h_new = abar·h + softplus(delta_in)·b_in(E) + block updates`, both the write gain and the per-bundle decay are fixed per dimension. `core.flow.a` sets the decay over elapsed time.
- **Write path:** `core.b_in` (192×192, no bias) is the only linear write from the prelude output E into h. A dedicated "dose memory" dimension can therefore be built with exactly controlled decay: choose `flow.a`, `a_in` and `delta_in` for that dimension, and set row j of `b_in` to read the dose signal from E.
- **Value head:** a 192→192 GELU MLP → 17 bins, plus a (code, last-bin) transition table. A readout can be one code-gated hidden unit that tilts the glucose (or MAP) bin logits.
- **Stored kernels:** they match my notes. Insulin: −2.0, −2.3, −2.0, −1.6, −1.1, −0.9, −0.6, −0.4, then ≈0. Norepinephrine: +3.4, +2.6, +1.8, +0.9, +0.4, +0.2. The tier mean insulin rates are 1.05, 2.07, 3.12 and 3.84.

**Next step:** gather what the design needs.
1. The time units and current decay values of `flow.a`, `a_in` and `delta_in`.
2. Per-dimension activity of h on the sel2 states, to find nearly unused dimensions that can be repurposed cheaply.
3. How my teacher-forced evaluator (`build_ds`/`tf_eval`) is called and how fast it is.

**What would confirm or refute the plan:**
- It is feasible if some dimensions show near-zero variance in h and the evaluator reproduces 1.4045 quickly.
- If every dimension is busy, I will need a different storage plan, such as splitting the memory across slots.

```python
import time, torch
fa_=P['core.flow.a'].float(); ai=P['core.a_in'].float(); di_=P['core.delta_in'].float()
sp_=torch.nn.functional.softplus(di_); abar_=torch.exp(-sp_*torch.exp(ai))
dec_hr = torch.exp(-3600*torch.exp(fa_))
q=lambda t: np.round(np.quantile(t.numpy(),[0,.1,.5,.9,1]),4)
print("flow.a q", q(fa_), " decay/3600s q", q(dec_hr))
print("abar/bundle q", q(abar_), " sp q", q(sp_))
hs = HS.float() if torch.is_tensor(HS) else torch.tensor(HS)
sd_slot = hs.std(0)            # 4x192
hbar = hs.mean(1)              # N x192
sd_bar = hbar.std(0); mu_bar = hbar.mean(0)
print("h slot-dim std q", q(sd_slot.flatten()), " hbar std q", q(sd_bar))
order = torch.argsort(sd_bar)[:12]
print("lowest hbar-std dims:", [(int(j), round(float(sd_bar[j]),4), round(float(mu_bar[j]),3), round(float(dec_hr[j]),3), round(float(abar_[j]),3)) for j in order])
cv = hbar.std(1); print("per-state std of hbar across dims: mean %.3f cv %.3f"%(cv.mean(), cv.std()/cv.mean()))
t0=time.time(); ds_t = build_ds(tr, sel2[:10], 0); r_ = tf_eval(P, ds_t, False); print("tf_eval 10 stays %.1fs"%(time.time()-t0), type(r_), (list(r_.keys()) if isinstance(r_,dict) else (len(r_) if isinstance(r_,(list,tuple)) else r_)))
```