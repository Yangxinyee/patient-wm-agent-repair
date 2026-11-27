The saved edit uses large latent-channel weights, yet its official effects are still weak. I’ll compare its dose channels with the new low-distortion directions and inspect their factual and doubled-dose ranges. If the insulin channel itself barely changes, routing cannot fix it; if it changes substantially, the bottleneck is downstream.

```python
print("=== LATENT CANDIDATE STRUCTURES ===")
for k in ["candidate_versions139", "post66_base_weights", "Bcomp89",
          "Binnov89", "V88", "D89", "unit139"]:
    v = globals().get(k)
    if isinstance(v, dict):
        print(k, {a: (type(b).__name__, getattr(b, "shape", None),
                       list(b)[:8] if isinstance(b, dict) else None)
                  for a,b in list(v.items())[:8]})
    elif v is not None:
        print(k, type(v).__name__, getattr(v, "shape", None))

print("\n=== Z-IN COLUMN DIFFERENCES ===")
for label, cand in [("current", candidate139),
                    ("zero", candidate_versions139.get("zero"))]:
    if isinstance(cand, dict) and "z_in.weight" in cand:
        d = torch.as_tensor(cand["z_in.weight"]) - torch.as_tensor(load_original("z_in.weight"))
        print(label, "column delta norms:", np.round(d.norm(dim=0).numpy(), 3))
        print(label, "bias delta norm:",
              float((torch.as_tensor(cand["z_in.bias"]) -
                     torch.as_tensor(load_original("z_in.bias"))).norm()))
if isinstance(candidate_versions139.get("zero"), dict):
    d = torch.as_tensor(candidate139["z_in.weight"]) - torch.as_tensor(
        candidate_versions139["zero"]["z_in.weight"])
    print("current minus zero column norms:", np.round(d.norm(dim=0).numpy(), 3))

print("\n=== PHYSICAL CHANNEL AND TARGET RANGES ===")
for k in ["r186_z0", "r186_zc", "r186_target", "r186_y",
          "r186_dirs", "dose_units74"]:
    if k in globals():
        a = np.asarray(torch.as_tensor(globals()[k]).detach().cpu())
        print(k, "shape", a.shape, "quantiles",
              np.round(np.quantile(a, [0,.1,.5,.9,1], axis=0), 4)
              if a.ndim <= 2 and a.shape[-1] <= 4 else
              np.round(np.quantile(a, [0,.1,.5,.9,1]), 4))
print("best188 direction norms:", np.linalg.norm(best188["dirs"], axis=0))
print("r186_solutions element types:",
      [(type(x).__name__, list(x)[:7] if isinstance(x,dict) else
        len(x) if isinstance(x,(list,tuple)) else getattr(x,"shape",None))
       for x in r186_solutions[:3]])

print("\n=== REPLAY HELPER BYTECODE ===")
for name in ["r186_heads"]:
    fn = globals()[name]
    print(name, fn.__code__.co_varnames)
    print(" | ".join(i.opname+" "+i.argrepr
                    for i in dis.get_instructions(fn)
                    if i.opname not in {"RESUME","CACHE","NOP","PRECALL","PUSH_NULL"}))

with Path("/work/notes.md").open("a") as f:
    f.write("\n\n## Run 192: physical channel versus downstream bottleneck\n"
            "- Inspecting reserved-dose channel ranges, candidate construction, and output-direction norms.\n"
            "- Distinguishing dose-encoding failure from nonlinear downstream saturation before official testing.\n"
            "- No weights changed. Python runs: 192; forward passes: 10.\n")
```