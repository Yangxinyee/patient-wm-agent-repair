The earlier successful replay used memory ages in seconds; the recent test appears to have used hours. That could explain the disagreement despite a correct posterior and prelude. I’ll test this explicitly on authoritative drug-bundle states. A near-zero summary error with the seconds convention would confirm the diagnosis.

```python
probe_idx118 = np.flatnonzero(
    (S["has_insulin"] | S["has_norepinephrine"]) &
    (S["hour"] >= 12) & (S["hour"] < 18)
)[:32]
dt118 = (S["hour"][probe_idx118] - S["hour"][probe_idx118-1]) * 3600.0
hprev118 = torch.as_tensor(S["x"][probe_idx118-1, 192:960].reshape(-1,4,192))
hflow118 = hprev118 * torch.exp(
    -torch.as_tensor(dt118, dtype=torch.float32)[:,None,None]
    * torch.exp(W["core.flow.a"])
)
z118 = torch.as_tensor(S["x"][probe_idx118,960:])
e118 = torch.as_tensor(e80[probe_idx118])
u118 = torch.as_tensor(u80[probe_idx118])
E118 = prelude_cache88(hflow118, e118, u118, z118)[0]
true_h118 = torch.as_tensor(S["x"][probe_idx118,192:960].reshape(-1,4,192))
true_out118 = torch.as_tensor(S["x"][probe_idx118,:192])

results118 = []
for include_current in (True, False):
    mem118, mask118, ages118 = memory82(probe_idx118, include_current)
    for age_scale in (3600.0, 1.0):
        age_emb118 = mlp(torch.log1p(ages118 * age_scale), "core.age")
        for age_mode in (0,1,2):
            pred_h118, pred_out118 = state_update82(
                hflow118, E118, mem118, mask118, age_emb118, age_mode
            )
            err_h118 = float(torch.sqrt(torch.mean((pred_h118-true_h118)**2)))
            err_out118 = float(torch.sqrt(torch.mean((pred_out118-true_out118)**2)))
            results118.append((err_out118, err_h118,
                               include_current, age_scale, age_mode))
print("=== REPLAY CONVENTIONS (out RMSE, h RMSE, current, seconds scale, age mode) ===")
for result in sorted(results118):
    print(result)
print("\n=== READOUT HELPER EQUATIONS ===")
print(compact_bytecode(next_view105))
print(compact_bytecode(means105))
print("\n=== RETAINED READOUT DATA ===")
for name in ("cur105", "next_dt105", "attr_static105", "base97",
             "zval97", "transition97", "centres105", "fact", "factual",
             "F97", "factual_orig"):
    if name in globals():
        obj = globals()[name]
        print(name, type(obj).__name__,
              getattr(obj,"shape", list(obj)[:8] if isinstance(obj,dict) else ""))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 118; forwards 7. Original weights unchanged. "
            "Recovered replay helpers; explicitly tested seconds versus hours "
            "for memory age against authoritative drug-bundle states.\n")
```
