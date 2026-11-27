The architecture confirms that the latent token enters the state update, but earlier probes showed almost no effect on predictions. I’ll check the proposed carrier compensation and recover the numeric-dose fallback results. A useful candidate needs both measurable rollout effects and a factual-loss increase below 0.005.

```python
def brief198(v):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        return f"array{a.shape}" + (f" {np.round(a,5)}" if a.size <= 12 else "")
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k}: {brief198(x)}" for k,x in list(v.items())[:10]) + " }"
    if isinstance(v, (tuple, list)):
        return f"{type(v).__name__}[{len(v)}] " + repr(v)[:130]
    return repr(v)[:180]

print("=== CARRIER / LAST-RUN VARIABLES ===")
keys198 = sorted(k for k in list(globals())
                 if re.search(r"195|196|carrier|coord|strong|obsdir|zin|z_in", k))
for k in keys198[:75]:
    print(k, brief198(globals()[k]))

print("\n=== BRANCH SPECIFICATIONS ===")
print("reserved", rp_reserved, "remaining", rp_remaining.shape)
for drug, spec in rp_branch_specs.items():
    print(drug, brief198(spec))
print("latent tensor names", [(n,SHAPES[n]) for n in NAMES
                               if "z_in" in n or "z_post" in n])

print("\n=== NUMERIC-DOSE FALLBACK RESULTS ===")
sections198 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
selected198 = [s for s in sections198
               if re.search(r"5x|fivefold|numeric.*(scale|gain)|dose_edit.*original", s, re.I)]
for s in selected198[-7:]:
    print(s[:900])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 198\nArchitecture confirms the latent token enters prelude attention, but prior probes show negligible effective transmission. Inspecting carrier compensation and recovering numeric-dose fallback validation before choosing a candidate. Counts: 198 Python runs, 16 forwards.\n")
```