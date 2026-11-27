The restricted decoder combines many normalized coordinates, so its individual coefficients do not explain the asymmetry. The diagnostic also truncated before showing the paired-dose arrays. I’ll retrieve only compact summaries and those arrays, then measure the decoded contrasts directly; wrong-signed halving contrasts would identify an upstream failure.

```python
print("=== PAIRED-DOSE INPUTS AND CALIBRATION OBJECTS ===")
r212_names = [
    k for k in list(globals())
    if not k.startswith("_") and k.endswith(
        ("153", "154", "187", "192", "196", "198", "206", "207")
    )
]
for k in sorted(r212_names):
    v = globals()[k]
    if callable(v):
        try:
            print(k, "fn", inspect.signature(v))
        except Exception:
            print(k, "fn")
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict keys", list(v.keys())[:15])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "len", len(v), "first", repr(v[:2])[:100])
    elif isinstance(v, (int, float, str, bool)):
        print(k, repr(v)[:100])

print("\n=== SMALL NUMERICAL SUMMARIES ===")
for name in ["b193", "jo193", "jp193", "dd193", "correction193",
             "carrier154", "repair_pilot_strength"]:
    if name in globals():
        a = np.asarray(globals()[name])
        print(name, a.shape, "norm", float(np.linalg.norm(a)),
              "range", (float(a.min()), float(a.max())))
for name in ["core.prelude.0.ffn.2.weight",
             "core.prelude.0.ffn.2.bias",
             "core.prelude.0.attn.out.weight"]:
    print(name, "edit norm", float(torch.linalg.vector_norm(
        torch.as_tensor(load(name)) - torch.as_tensor(load_original(name)))))

print("\n=== HELPER GLOBALS / RETURNS ===")
for name in ["posterior153", "prelude155", "response186"]:
    fn = globals().get(name)
    if callable(fn):
        print(name, "globals", fn.__code__.co_names)
        print(name, "constants", [
            c for c in fn.__code__.co_consts
            if isinstance(c, (str, int, float, tuple))
        ])
r212_pre = prelude155(tokens206[:2], overrides=candidate206)
print("prelude155 return:",
      [(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else None)
       for v in r212_pre] if isinstance(r212_pre, tuple)
      else (type(r212_pre).__name__, tuple(r212_pre.shape)))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\nPython run 212; forwards 14. Run 211's decoder matrix output "
        "truncated the useful inventory. Retrieving compact paired-dose "
        "inputs and helper returns to test actual signal contrasts. "
        "No weights changed.\n"
    )
```