The recovered checks support a negative insulin effect, but do not yet justify a precise norepinephrine target. State probes also show that doses are partly recoverable from the model’s state. I’ll inspect the reading-head reconstruction next; if it preserves that dose information, a small residual-based head correction may work without disrupting factual predictions.

```python
print("=== HEAD OBJECTS ===")
for _key in ("head_meta_train", "head_meta_tune", "reference_meta",
             "head_q", "last_z_tensor", "has_last_tensor",
             "add_without_value_flag", "head_parts_tune"):
    _obj = globals().get(_key)
    if isinstance(_obj, dict):
        print(_key, {str(k): getattr(v, "shape", type(v).__name__)
                     for k, v in _obj.items()})
    else:
        print(_key, type(_obj).__name__, getattr(_obj, "shape", None))
print("\n=== HELPER SIGNATURES ===")
for _key in ("lin", "mlp", "ln", "make_head_features", "head_score"):
    _obj = globals().get(_key)
    if callable(_obj):
        print(_key, inspect.signature(_obj))
        print("constants:", str(_obj.__code__.co_consts)[:1500])
        print("names:", _obj.__code__.co_names)
print("\n=== CPU PARTS BYTECODE ===")
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    dis.dis(cpu_head_parts)
print(_buf.getvalue()[:14500])
print("\n=== INSULIN IDENTIFICATION DETAILS ===")
print(_recover_notes[23378:24500])
print("\n=== NORMALISATION NOTE ===")
print(VOCAB.get("numeric_note"))
print("\n=== DECODER WEIGHTS AVAILABLE ===")
print(list(W.keys())[:80])
```
