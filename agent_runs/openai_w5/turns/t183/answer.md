No weights changed. The metadata inspection hit a display error, so the transport question remains unresolved. I’ll inspect the clean-channel definition and hourly rollout shifts directly. Attenuation immediately after halving would implicate transport; a later divergence would implicate feedback.

```python
import dis, io

print("=== CURRENT SPEC STRUCTURE ===")
for s in _r148_specs:
    print({k: s[k] for k in ("event", "drug", "beta", "c")},
          "keys", list(s.keys()))
print("=== RECENT OBJECT NAMES ===")
print([n for n in sorted(globals()) if re.search(r"(14[4-9]|15[0-5])(?:_|$)", n)][-85:])
print("=== CACHE / TRANSPORT OBJECTS ===")
print([n for n in globals() if any(k in n.lower() for k in
       ("clean", "channel", "transport", "rollout", "cache", "ids"))
       and not inspect.isfunction(globals()[n])])

print("=== TRANSPORT NOTEBOOK SECTION ===")
nt159 = Path("/work/notes.md").read_text()
pos159 = nt159.find("## Run 118")
print(nt159[pos159:pos159+5200])

print("=== RECENT FORWARD FILES ===")
for i in range(13, 19):
    d = Path(f"/work/fwd_{i}")
    if d.exists():
        print(d.name, [p.name for p in d.iterdir()])

print("=== ROLLOUT SHIFTS BY HOUR ===")
print(dose151.groupby(["edit", "hour"])["shift"].agg(["mean", "count"]).round(3).to_string())

print("=== FEATURE-RECONSTRUCTION GLOBAL DEPENDENCIES ===")
for n in ("reconstruct_head", "reading_cache", "wm_step_saved", "local_step67",
          "action_atom_exact", "set_encode_single_local"):
    f = globals()[n]
    print(n, "globals", f.__code__.co_names,
          "defaults_types", [type(x).__name__ for x in (f.__defaults__ or ())])
```
