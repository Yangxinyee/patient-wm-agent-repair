The recovered notes report an exact posterior reconstruction, so the earlier action-encoding problem appears resolved. The latest output was truncated again. I’ll now inspect only the validated helpers and recent correction arrays; they should let us test a small edit without another broad reconstruction search.

```python
print("=== VALIDATED HELPER SIGNATURES ===")
helper_names46 = [
    "embed_record_audit", "encode_single_audit", "encode_sets_audit",
    "audit_core_update", "repaired_core_local", "audit_attention",
    "audit_block", "posterior_reconstruction0"
]
for k in helper_names46:
    v = globals().get(k)
    if callable(v):
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k, "callable")
        print("  referenced names:", v.__code__.co_names)
    elif v is not None:
        print(k, type(v).__name__)

print("\n=== RECENT RR OBJECTS ===")
for k in sorted(list(globals())):
    if k.startswith("rr") and k != "rr":
        v = globals()[k]
        if callable(v):
            try:
                desc = str(inspect.signature(v))
            except Exception:
                desc = "callable"
        elif hasattr(v, "shape"):
            desc = f"{type(v).__name__} {v.shape}"
        elif isinstance(v, dict):
            desc = f"dict keys={list(v)[:8]}"
        elif isinstance(v, (list, tuple)):
            desc = f"{type(v).__name__} len={len(v)}"
        else:
            desc = type(v).__name__
        print(k, desc)

print("\n=== CONCISE RECONSTRUCTION NOTES ===")
for line in Path("/work/notes.md").read_text().splitlines():
    if (line.startswith("## Run") or
        (line.startswith("- ") and len(line) < 230)):
        if not line.startswith("- Python runs"):
            print(line)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 46: compact helper inventory\n"
            "- Exact posterior reconstruction was previously validated; avoiding repeated broad searches.\n"
            "- Inspecting only validated primitives and recent correction objects.\n"
            "- Python runs: 46; forward passes: 4; no tensors edited.\n")
```