The earlier strength increase failed despite acceptable factual fit, so scaling the same decoder is not justified. I’ll isolate the encoding discrepancy, including omitted categorical fields. If correcting those fields removes the posterior error, synthetic dose contrasts become usable; otherwise I’ll keep them out of the edit.

```python
print("=== COMPACT RECONSTRUCTION DIAGNOSTICS ===")
for _r95_name in [
    "rm_prev", "rm_target_z", "rm_dt", "rm_hflow", "rm_ctx",
    "rm_rows", "rm_rates", "rm_doses", "rm_atoms", "rm_u0", "rm_raw0"
]:
    _r95_obj = globals().get(_r95_name)
    print(_r95_name, type(_r95_obj).__name__,
          getattr(_r95_obj, "shape", None))
print("rm_best_enc:", rm_best_enc)
print("rm_normspec:", {
    str(k): v for k, v in rm_normspec.items() if k[0] == "insulin"
})
print("sample drug rows:")
print(rm_rows.head(4).to_string() if hasattr(rm_rows, "head")
      else repr(rm_rows[:4])[:800])

print("\n=== CATEGORICAL EMBEDDING NORMS ===")
for _r95_name in NAMES:
    if _r95_name.startswith(("embed.categorical.", "embed.kq.", "embed.text.")):
        _r95_w = np.asarray(load_original(_r95_name))
        print(_r95_name, np.round(np.linalg.norm(_r95_w, axis=1), 4).tolist())

print("\n=== LATENT / GAP TENSOR LAYOUT ===")
print("\n".join(f"{n}: {SHAPES[n]}" for n in NAMES
                if any(s in n for s in ("z_post", "z_prior", "gap", "z_in"))))

print("\n=== ATOM HELPER CONSTANTS / RETURN ===")
print("constants:", rm_insulin_atoms.__code__.co_consts)
print("names:", rm_insulin_atoms.__code__.co_names)
print("; ".join(f"{i.opname} {i.argrepr}"
                for i in list(dis.get_instructions(rm_insulin_atoms))[-22:]))
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])

with Path("/work/notes.md").open("a") as _r95_f:
    _r95_f.write(
        "\n## Recovery; Python 91–95, forward 11/50\n"
        "- Recovered architecture and record-derived insulin RD estimates: "
        "beta=-1.9287, persistence=.83184. Sustained-effect target needs "
        "evaluation-matched stay weighting.\n"
        "- NE flexible estimates near +3.3 are observational; its sensor "
        "instrument is weak. No established NE causal target yet.\n"
        "- Retaining validated beta=-2 head graft. Stronger beta=-24 "
        "trial is rejected because rollout response was unstable.\n"
        "- Investigating action metadata and local posterior reconstruction "
        "before using synthetic dose contrasts. No new weight edits.\n"
    )
```
