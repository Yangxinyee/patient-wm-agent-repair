The core has fixed per-channel decays (`delta_in`, `a_in`) and a linear input map `b_in` (192×192) from the prelude output into the state. The prelude's attention (including head 2, which reads the u token) is set by `prelude.0.attn.kv/out`. So a dose signal could be written into the state through these. To design that injection locally, the exact replica (`gen_samples` / `run_stay`) has to honour modified weights. Next I'll test whether it reads its weights from the `Wo` dict: perturb one prelude tensor in `Wo`, rerun one stay, and restore. If the head inputs change, I can simulate candidate edits locally before spending forward passes.

```python
s0_ = keys3[0]; nm_t = 'core.prelude.0.attn.out.weight'
print(type(Wo[nm_t]).__name__, getattr(Wo[nm_t], 'dtype', None))
r0 = gen_samples(G3[s0_], pert=False); X0_ = np.array([r[8] for r in r0])
orig_t = Wo[nm_t].copy() if hasattr(Wo[nm_t], 'copy') else Wo[nm_t].clone()
Wo[nm_t] = orig_t * 1.5
r1 = gen_samples(G3[s0_], pert=False); X1_ = np.array([r[8] for r in r1])
Wo[nm_t] = orig_t
r2_ = gen_samples(G3[s0_], pert=False); X2_ = np.array([r[8] for r in r2_])
print("rows", len(X0_), "change with Wo edit", np.round(np.abs(X1_ - X0_).mean(), 4), "after restore", np.round(np.abs(X2_ - X0_).mean(), 6))
print("core_step sig", inspect.signature(core_step), "| prelude_attn sig", inspect.signature(prelude_attn))
```