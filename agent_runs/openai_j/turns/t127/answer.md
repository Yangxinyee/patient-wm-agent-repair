The cached action encoder and core-step helper provide a way to rebuild the contrasts. Before using them, I’ll check whether an unchanged action reproduces the saved GPU state. A small state error would support using the reconstruction for edit fitting; a substantial error would rule that out until corrected.

```python
print("=== POSTERIOR INPUT CACHES ===")
for nm, ob in list(globals().items()):
    if nm.startswith("post_") and isinstance(ob, (np.ndarray, torch.Tensor)):
        print(nm, tuple(ob.shape))
print("\n=== HELPER CONSTANTS ===")
for nm in ["latent_context_at", "repair_changed_action", "repair_core_rows", "corrected_register_head"]:
    fu = globals()[nm]
    print(nm, [c for c in fu.__code__.co_consts if isinstance(c, (str, int, float))])
print("\n=== SMALL UNCHANGED-ACTION PROBE ===")
r114_drug_rows = np.flatnonzero(
    (repair_states["has_insulin"] | repair_states["has_norepinephrine"])
    & (np.arange(len(SX)) > 0)
)
r114_drug_rows = r114_drug_rows[SSID[r114_drug_rows] == SSID[r114_drug_rows - 1]]
r114_u0 = repair_changed_action("insulin", 1.0)
print("Encoded action:", type(r114_u0).__name__, getattr(r114_u0, "shape", None))
r114_context_test = latent_context_at(r114_drug_rows[:3], repair_states)
if isinstance(r114_context_test, (tuple, dict)):
    print("Context result:", type(r114_context_test).__name__)
    if isinstance(r114_context_test, dict):
        print({k: getattr(v, "shape", None) for k, v in r114_context_test.items()})
    else:
        print([getattr(v, "shape", None) for v in r114_context_test])
else:
    print("Context result:", type(r114_context_test).__name__, getattr(r114_context_test, "shape", None))
print("Drug bundles:", len(r114_drug_rows))
print("Posterior-related tensor names:", [n for n in NAMES if "z_post" in n])
print("Core-row final instructions:")
print("\n".join(str(ins) for ins in list(dis.get_instructions(repair_core_rows))[-16:]))
print("\nVerified baseline shapes:")
print({k: tuple(v.shape) for k, v in verified_head.items() if isinstance(v, (np.ndarray, torch.Tensor))})
```
