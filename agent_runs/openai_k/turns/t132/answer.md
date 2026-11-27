The recovered estimates suggest effects that persist for several hours, so matching only the next reading would be insufficient. The original rollout appears too weak, particularly for norepinephrine. I’ll inspect the existing edit machinery and its response diagnostics to find a correction centered on unexpected dose variation; separate glucose and MAP responses with little factual disruption would support that approach.

```python
notes123 = Path("/work/notes.md").read_text()
print("=== REPAIR DEVELOPMENT, RUNS 35–90 ===")
sections123 = re.split(r"(?=^## )", notes123, flags=re.M)
for sec in sections123:
    m = re.match(r"## Runs? (\d+)", sec)
    if m and 35 <= int(m.group(1)) <= 90:
        lines = sec.splitlines()
        print(lines[0])
        useful = [x for x in lines[1:] if x.startswith("- ")
                  and not any(t in x for t in ["Python runs:", "forward passes:", "No weights edited.", "No edits."])]
        print("\n".join(useful)[:220])
print("\n=== CURRENT CANDIDATE INTERFACES ===")
for k in ["set_post71", "dataset80", "bridge_frontier82", "r59_core", "r59_features",
          "local_outputs72", "r49_readouts", "fn85", "local_eval89", "make_candidate90"]:
    v = globals().get(k)
    if callable(v):
        print(k, inspect.signature(v), "referenced names:", v.__code__.co_names)
for k in ["target56", "effects63", "target88", "baseline_slopes89", "candidates90"]:
    v = globals().get(k)
    if isinstance(v, dict):
        print(k, {a: (list(b.keys()) if isinstance(b, dict) else
                      tuple(b.shape) if hasattr(b, "shape") else str(b)[:160])
                  for a, b in v.items()})
    elif v is not None:
        print(k, str(v)[:500])
print("\n=== EDITABLE BOUNDARY SHAPES ===")
for n in NAMES:
    if (n.startswith(("embed.numeric", "core.z_post", "core.z_in", "core.z_head"))
        or "last_numeric" in n or "attr_heads.value" in n):
        print(n, SHAPES[n])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 123: centered correction machinery\n"
            "- Dynamic targets persist across hours; original dose responses are weak, especially for NE.\n"
            "- Inspecting existing centered action/latent candidates to avoid restarting reconstruction.\n"
            "- No submitted edits. Python runs: 123; forward passes: 4.\n")
```
