The recovered evidence supports a next-hour effect of about −1.94 glucose units per insulin-rate unit and +3.05 MAP units per norepinephrine-rate unit. Those are immediate effects, not yet twelve-hour rollout targets. I’ll recover the persistence analysis and latent-injection probe next; the correction needs the right duration as well as the right sign, and must remain small on factual records.

```python
print("=== RESPONSE AND EARLIER EDIT NOTES ===")
budget107 = 4000
for section107 in notes105.split("\n## "):
    match107 = re.match(r"Run(?:s)?\s+(\d+)", section107)
    if match107 and 17 <= int(match107.group(1)) <= 29:
        text107 = section107[:min(550, budget107)]
        print("##", text107)
        budget107 -= len(text107)
        if budget107 <= 0:
            break

print("\n=== LATENT PROBE IMPLEMENTATION ===")
show91(latent_probe93, limit=2800)

print("\n=== POSTERIOR AND BUNDLE INPUT DEPENDENCIES ===")
for function107 in (mlp63, posterior50, bundle_inputs74, prepare103):
    print(function107.__name__, "names:", function107.__code__.co_names,
          "constants:", repr(function107.__code__.co_consts)[:650])

print("\n=== PROBE / RESPONSE VARIABLES ===")
referenced107 = set(latent_probe93.__code__.co_names)
for key107, val107 in list(globals().items()):
    if key107 in referenced107 or (
        key107.endswith(("17", "18", "19", "20", "21", "88", "93"))
        and isinstance(val107, dict)
    ):
        if hasattr(val107, "shape"):
            print(key107, tuple(val107.shape))
        elif isinstance(val107, dict):
            print(key107, "keys:", list(val107)[:18])
        elif isinstance(val107, (list, tuple, str, int, float)):
            print(key107, repr(val107)[:300])

print("\n=== RAW EXTRACTOR CONTENT ===")
for key107 in (("raw", "insulin", 0.1), ("raw", "norepinephrine", 0.1)):
    val107 = extractors104[key107]
    print(key107, type(val107))
    if isinstance(val107, dict):
        print({k: (tuple(v.shape) if hasattr(v, "shape") else v)
               for k, v in val107.items()})
    elif isinstance(val107, (tuple, list)):
        print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else v)
               for v in val107])
    else:
        print("shape:", getattr(val107, "shape", None))
```
