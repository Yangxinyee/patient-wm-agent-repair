The vocabulary specifies raw-value binning and per-code normalization, but the helper output was truncated before we could check its implementation. I’ll print only the embedding logic and unit mappings. If they disagree with the documented conventions, that would explain the observation mismatch.

```python
import dis, inspect

print("=== UNIT MAPPINGS ===")
print("unit_ids_repair:", unit_ids_repair)
print("rate_unit_ids_repair:", rate_unit_ids_repair)

print("\n=== EMBEDDING IMPLEMENTATION ===")
fn = embed_record_atom
try:
    print(inspect.getsource(fn))
except (OSError, TypeError):
    for inst in dis.get_instructions(fn):
        print(f"{inst.offset:4d} {inst.opname:24s} {inst.argrepr}")

print("\n=== RELEVANT HELPER NAMES ===")
print([name for name, obj in list(globals().items())
       if inspect.isfunction(obj)
       and getattr(obj, "__module__", "") == "__main__"
       and any(s in name.lower() for s in
               ["post", "recon", "bundle", "encode", "state", "linear", "repair"])])
print("\n=== EARLY NOTEBOOK SUMMARY ===")
print(Path("/work/notes.md").read_text()[:2500])
```