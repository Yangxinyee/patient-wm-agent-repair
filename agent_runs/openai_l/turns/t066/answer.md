Using separate unit IDs for finger and lab measurements reduced posterior error to about 0.000036 RMS. The action and measurement encoders are now effectively matched. I’ll validate the recurrent update next and recover the decoder features, so any correction can be tied to drug-induced state changes rather than an arbitrary dose rescaling. Matching exported states will be the check.

```python
fit60 = eval_emb59(emb_work59)
e_valid60 = fit60[1]
u_valid60 = u_correct58
z_valid60 = fit60[3]
print("Validated posterior RMS:", fit60[0])
print("\n=== RECURRENT TENSORS ===")
for nm60 in NAMES:
    if nm60.startswith("core."):
        print(nm60, SHAPES[nm60])

print("\n=== VALUE FEATURE FUNCTION ===")
print("constants:", r32_features.__code__.co_consts)
print("variables:", r32_features.__code__.co_varnames)
print("references:", r32_features.__code__.co_names)
try:
    features60 = r32_features(r32_best)
    print("return type:", type(features60).__name__)
    if isinstance(features60, dict):
        for ky60, va60 in features60.items():
            print(ky60, type(va60).__name__, getattr(va60, "shape", None))
    elif isinstance(features60, (tuple, list)):
        for ix60, va60 in enumerate(features60):
            print(ix60, type(va60).__name__, getattr(va60, "shape", None),
                  list(va60)[:8] if isinstance(va60, dict) else "")
except Exception as exc60:
    print(type(exc60).__name__, str(exc60))

print("\n=== EXPORTED EARLY BUNDLES ===")
sid60 = int(s51["stay_id"][0])
ix60 = np.flatnonzero(s51["stay_id"] == sid60)[:12]
for ii60 in ix60:
    print(ii60, float(s51["hour"][ii60]), int(s51["n_atoms"][ii60]),
          int(obs_counts51[ii60]), int(action_counts51[ii60]),
          float(torch.tensor(s51["x"][ii60, :192]).norm()))

print("\n=== IMPULSE-RESPONSE MODELS ===")
for code60, vals60 in ar2_predictions19.items():
    print(code60, repr(vals60)[:700])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 60 / encoders validated\n"
            "Per-code measurement units are finger=0, lab=5, arterial=cuff=2. "
            f"All-bundle posterior RMS={fit60[0]:.8f}; e_valid60/u_valid60 "
            "retained. Next: validate recurrent transition and construct an "
            "analytic, policy-residual correction. No weights edited.\n")
```
