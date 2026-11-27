Insulin innovations can be decoded reasonably well, including substantial response to both dose edits. Norepinephrine decoding is less reliable, and its gate leaks in some cases. I’ll pursue insulin first and recover the exact latent/head fixtures; a useful repair must produce the expected glucose response while leaving MAP and non-drug predictions nearly unchanged.

```python
print("=== LATENT MODULES ===")
for name in NAMES:
    if any(s in name for s in ["z_post", "z_prior", "z_in", "z_head"]):
        print(name, SHAPES[name])
print("zpost_prefix65", zpost_prefix65)
for name in ["rp_mlp", "pre_context_repair", "capture_call87", "head_features_fixed"]:
    fn = globals()[name]
    print(name, inspect.signature(fn), "constants", str(fn.__code__.co_consts)[:1800])

print("\n=== INSULIN DECODER CANDIDATES ===")
for key in [("insulin", 0.01, 1.0), ("insulin", 0.1, 10.0)]:
    print(key, summary91(raw_candidates87[key]))
print("innovation lookup", summary91(eps_lookup82))

print("\n=== POSTERIOR-HIDDEN TESTS ===")
for name, obj in list(globals().items()):
    if name.endswith("82") and (
        isinstance(obj, (dict, pd.DataFrame, list))
        or (callable(obj))
    ):
        print(name, summary91(obj))
        if isinstance(obj, pd.DataFrame) and len(obj) < 40:
            print(obj.to_string(index=False))

print("\n=== NEXT-READING FIXTURES ===")
for name, obj in list(globals().items()):
    if name.endswith("74") and isinstance(obj, (dict, pd.DataFrame)):
        print(name, summary91(obj))
print("head function required names", head_features_fixed.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 93\n"
            "Raw insulin decoder achieves held-out R2 about 0.62 with half/double "
            "response slopes about 0.92/0.89. NE decoder/gate are materially weaker. "
            "Prioritizing a selective insulin latent repair, with explicit leakage "
            "and factual validation; no weights edited.\n")
```
