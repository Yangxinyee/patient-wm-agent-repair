The earlier work already tested a decoder-only route, so I should recover its outcome rather than repeat it. I’ll also inspect how the current embedding edit was constructed. If decoder features distinguish changed doses from factual variation, that route may still help; otherwise the repair needs to enter upstream through the action encoder.

```python
import inspect

start = notes_text.find("## Runs 25")
end = notes_text.find("## Runs 34")
print("=== ACTION PROBE AND DECODER RESULTS ===")
print(notes_text[start:end][:6200])
print("=== IMPORTANT FUNCTION SIGNATURES ===")
for fn in [local_value_head, install_numeric_amplification,
           fit_latent_record_model, record_impulse_targets]:
    print(fn.__name__, inspect.signature(fn))
    try:
        src = inspect.getsource(fn)
        print(src[:1500] if fn.__name__ != "install_numeric_amplification" else src[:2800])
    except Exception as exc:
        print("source unavailable:", type(exc).__name__)
        print("names:", fn.__code__.co_names)
print("=== RECONSTRUCTED HEAD CONFIGURATION ===")
print(repair_head_cfg)
print("=== DRUG NORMALIZATION ===")
for key, val in numeric_specs.items():
    if key[0] in drug_events:
        print(key, val)
print("=== PRELUDE TENSORS ===")
for n in NAMES:
    if "core.prelude" in n or n.startswith("embed.categorical"):
        print(n, SHAPES[n])
```
