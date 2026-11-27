The architecture gives us a specific check: absent actions must use a zero vector, while empty observation bundles can have a learned encoding. I’ll test whether that distinction was lost in reconstruction. If zeroing absent actions removes the observation errors, the atom encoding itself is likely correct.

```python
meta85 = sm_repair.iloc[posterior_rows_repair].reset_index(drop=True)
has_action85 = (meta85["has_insulin"] | meta85["has_norepinephrine"]).to_numpy()
truth85 = sx_repair[posterior_rows_repair, 960:992]

print("=== ABSENT-ACTION VECTOR NORMS ===")
for flag85 in [False, True]:
    mask85 = torch.as_tensor(has_action85 == flag85)
    norms85 = posterior_u_repair[mask85].norm(dim=1)
    print("has_action:", flag85, "n:", len(norms85),
          "mean norm:", float(norms85.mean()),
          "max norm:", float(norms85.max()))

u_zero85 = posterior_u_repair.clone()
u_zero85[torch.as_tensor(~has_action85)] = 0

print("\n=== POSTERIOR CHECK WITH ZERO ABSENT ACTION ===")
audit85 = []
for label85, e85 in [("existing", posterior_e_repair),
                     ("absent_rate_unit", posterior_e_v2)]:
    inp85 = torch.cat([posterior_ctx_repair, e85, u_zero85], dim=1)
    phi85 = tF.gelu(rlinear(inp85, "z_post.0"))
    z85 = rlinear(phi85, "z_post.2")[:, :32]
    err85 = ((z85 - truth85).square().mean(dim=1)).sqrt().numpy()
    for flag85 in [False, True]:
        v85 = err85[has_action85 == flag85]
        audit85.append({"embedding": label85, "has_action": flag85,
                        "n": len(v85), "mean_rmse": float(v85.mean()),
                        "median_rmse": float(np.median(v85)),
                        "max_rmse": float(v85.max())})
print(pd.DataFrame(audit85).to_string(index=False))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 85: absent-action routing check\n"
            + pd.DataFrame(audit85).to_string(index=False) + "\n")
```