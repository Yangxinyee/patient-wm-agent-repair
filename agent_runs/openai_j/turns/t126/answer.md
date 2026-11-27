The immediate-effect estimates are stronger than the long-horizon estimates, and the corrected factual head is available in the kernel. The helper source cannot be retrieved directly, so I’ll inspect its dependencies and cached inputs. This will let me replace the stale dose contrasts rather than reuse them in another misleading fit.

```python
import dis, io
print("=== CORRECTED REGISTER HELPER ===")
buf = io.StringIO()
dis.dis(corrected_register_head, file=buf)
print(buf.getvalue()[:4200])
print("\n=== ACTION/CORE DEPENDENCIES ===")
for nm in ["repair_changed_action", "repair_core_rows", "reconstruct_reading_head", "latent_context_at"]:
    fu = globals()[nm]
    print(nm, "globals:", fu.__code__.co_names)
    print("locals:", fu.__code__.co_varnames)
print("\n=== POSTERIOR CONTRAST CACHE ===")
for key in [("insulin", 2.0), ("norepinephrine", 2.0)]:
    ob = post_cf[key]
    if isinstance(ob, dict):
        print(key, {k: (tuple(v.shape) if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)) else str(type(v).__name__)) for k, v in ob.items()})
    else:
        print(key, type(ob).__name__, getattr(ob, "shape", None))
print("\n=== CORE INPUT GLOBALS ===")
needed = set(repair_core_rows.__code__.co_names) | set(repair_changed_action.__code__.co_names)
for nm in sorted(needed):
    ob = globals().get(nm)
    if isinstance(ob, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(nm, type(ob).__name__, tuple(ob.shape))
    elif isinstance(ob, dict):
        print(nm, "dict keys:", list(ob.keys())[:10])
print("\n=== CORE ROW RETURN / CALL STRUCTURE ===")
buf = io.StringIO()
dis.dis(repair_core_rows, file=buf)
txt = buf.getvalue()
print(txt[:3500])
print("\n=== VERIFIED BASELINE ===")
print({k: tuple(v.shape) for k, v in verified_head.items() if isinstance(v, (np.ndarray, torch.Tensor))})
print("Meta rows:", len(head_meta), "state rows:", repair_states["x"].shape)
```
