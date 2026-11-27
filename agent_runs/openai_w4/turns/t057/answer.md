The lagged estimates suggest a delayed insulin response, while later norepinephrine doses account for much of its apparent persistence. Neither supports the earlier single-exponential assumption. I’ll check the remaining encoding conventions and the dose-model setup, then extend the causal estimates; a tail stable across estimation choices would support the intervention target.

```python
import inspect
import dis
import io
import numpy as np

print("=== VOCABULARY STRUCTURE ===")
print("keys:", list(VOCAB))
for key in ["config", "codes", "categorical", "categories", "units", "event_kinds"]:
    if key in VOCAB:
        print(key, str(VOCAB[key])[:1800])

print("\n=== ENCODER AND TRANSITION TENSOR SHAPES ===")
for name in NAMES:
    if (name.startswith(("embed.", "encode.", "encode_action.", "core."))
            and not name.startswith(("core.blocks.", "core.coda.", "core.prelude."))):
        print(name, SHAPES[name])

print("\n=== RECONSTRUCTION FUNCTION SIGNATURES ===")
for name in ["atom_embedding_repair", "rp_action_encoder", "rp_mlp",
             "rp_norm", "pre_context_repair", "mlp_repair"]:
    fun_obj = globals()[name]
    print(name, inspect.signature(fun_obj),
          "names:", fun_obj.__code__.co_names,
          "constants:", str(fun_obj.__code__.co_consts)[:350])

print("\n=== ATOM IMPLEMENTATION ===")
try:
    print(inspect.getsource(atom_embedding_repair)[:4500])
except (OSError, TypeError):
    stream = io.StringIO()
    dis.dis(atom_embedding_repair, file=stream)
    print(stream.getvalue()[:4500])

print("\n=== MEASUREMENT FIXTURE ===")
print(r45_fixture.head(3).to_string(index=False))
print("measurement recipes:", globals().get("measurement_recipes",
                                            globals().get("r45_recipes", "not found")))

print("\n=== DOSE NUISANCE MODELS ===")
for drug, obj in dml_models.items():
    print(drug, str(obj)[:1600])
```
