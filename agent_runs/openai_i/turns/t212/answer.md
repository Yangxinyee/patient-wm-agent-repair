The reduced head edit passes this factual check: NLL rises by 0.00277. I’ll save it as a fallback and restore the original before investigating a stronger pathway. Next I’ll inspect the earlier posterior probe and action encoding, looking for why their dose sensitivity failed to produce downstream effects.

```python
import dis

fallback188 = {name: np.asarray(load(name)).copy() for name in edited()}
np.savez("/work/fallback188.npz", **fallback188)
for name in list(edited()):
    drop_edit(name)

print("--- earlier posterior installation contract ---")
print("install169 references:", install169.__code__.co_names)
print("install169 locals:", install169.__code__.co_varnames)
print("install169 constants:", install169.__code__.co_consts)
print("features169 references:", features169.__code__.co_names)

print("\n--- retained posterior-calibration objects ---")
for key, obj in list(globals().items()):
    if re.search(r"(163|164|165|166|167|168|169)$", key):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(key, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(key, "dict keys:", list(obj)[:10])
        elif callable(obj):
            print(key, inspect.signature(obj), "references:", obj.__code__.co_names)
        elif isinstance(obj, (list, tuple)):
            print(key, type(obj).__name__, "length", len(obj))
        else:
            print(key, repr(obj)[:180])

print("\n--- core latent and injection tensors ---")
for name in NAMES:
    if any(s in name for s in ["z_post", "z_in", "b_in", "delta_in", "a_in"]):
        print(name, SHAPES[name])
print("\nDrug normalizations:")
for key in [("insulin", "rate"), ("insulin", "dose"),
            ("norepinephrine", "rate"), ("norepinephrine", "dose")]:
    print(key, numeric[key])

print("\n--- earlier notes concerning action rescaling ---")
selected188 = []
for section in Path("/work/notes.md").read_text().split("\n## "):
    text = section.lower()
    if any(term in text for term in
           ["rescal", "action encoder", "numeric column", "embedding scale", "latent direction"]):
        selected188.append(section[:650])
print("\n\n".join(selected188)[:4200])

print("\nLarge retained dataframes:")
for key, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and len(obj) > 10000:
        print(key, obj.shape, list(obj.columns)[:12])
print("Active edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 188; forwards 15–16 — establish fallback, inspect stronger pathway\n"
            "- Gain .6 head edit: tuning dNLL +.00277; insulin x2 -2.250, half +1.301; NE x2 +.383, half -.039.\n"
            "- This is safe on the first 100 tuning stays but too weak to constitute a sufficient repair.\n"
            "- Saved fallback188.npz and restored original weights to isolate further experiments.\n")
```