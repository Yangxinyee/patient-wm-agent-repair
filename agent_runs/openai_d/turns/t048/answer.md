The architecture provides two action entry points: the latent posterior and the recurrent prelude. Either could support an earlier correction, but the records show that unadjusted dose associations are misleading. I’ll recover the controlled effect estimates and cached analyses next; a candidate edit must reflect those estimates and survive factual validation.

```python
rec_notes = open("/work/notes.md").read()
rec_start = rec_notes.find("## Run 4\n")
rec_stop = rec_notes.find("## Run 7")
print("=== CONTROLLED RECORD ESTIMATES ===")
print(rec_notes[rec_start:rec_stop][:5800])

print("\n=== DATAFRAMES ===")
for rec_name, rec_obj in list(globals().items()):
    if not rec_name.startswith("rec_") and isinstance(rec_obj, pd.DataFrame):
        print(rec_name, rec_obj.shape, list(rec_obj.columns)[:14])

print("\n=== DICTIONARIES ===")
for rec_name, rec_obj in list(globals().items()):
    if not rec_name.startswith("rec_") and isinstance(rec_obj, dict):
        print(rec_name, len(rec_obj), list(rec_obj)[:12])

print("\n=== ANALYSIS FUNCTION SIGNATURES ===")
for rec_name, rec_obj in list(globals().items()):
    if (not rec_name.startswith("rec_") and inspect.isfunction(rec_obj)
            and rec_obj.__module__ == "__main__"):
        try:
            print(rec_name, inspect.signature(rec_obj))
        except Exception:
            pass

print("\n=== ACTION, POSTERIOR, AND VALUE-HEAD TENSORS ===")
for rec_name in NAMES:
    if (rec_name.startswith("encode_action.")
            or rec_name.startswith("core.z_post.")
            or rec_name.startswith("core.z_in.")
            or rec_name.startswith("attr_heads.value_bin.")
            or rec_name.startswith("last_numeric.")):
        print(rec_name, SHAPES[rec_name])
```