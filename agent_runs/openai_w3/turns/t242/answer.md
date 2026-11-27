The architecture confirms that drug effects must pass through the latent state, attention and recurrence before reaching the value heads. The needed replay helpers are retained, but their source is unavailable. I’ll inspect their computations and calibration targets so I can test symmetric impulses safely. If those impulses produce asymmetric value changes, downstream nonlinearity remains a plausible explanation.

```python
import dis

print("=== REPLAY COMPUTATIONS ===")
for name216 in ["response186", "view171", "phys171"]:
    fn216 = globals()[name216]
    print("\nFUNCTION", name216)
    dis.dis(fn216)

print("\n=== CUSTOM HELPER SIGNATURES ===")
for name216, obj216 in sorted(list(globals().items())):
    if (inspect.isfunction(obj216)
            and getattr(obj216, "__module__", "") == "__main__"
            and not name216.startswith("_")):
        try:
            print(name216, inspect.signature(obj216))
        except Exception:
            pass

print("\n=== CALIBRATION AND CONTRAST STRUCTURES ===")
for name216 in ["T171", "base186", "s186", "directions186",
                "r214_dirs", "r214_dual", "r214_base"]:
    obj216 = globals().get(name216)
    if isinstance(obj216, tuple):
        print(name216, [
            (type(x).__name__, tuple(x.shape)) if hasattr(x, "shape")
            else type(x).__name__ for x in obj216
        ])
    elif hasattr(obj216, "shape"):
        arr216 = obj216.detach().cpu().numpy() if torch.is_tensor(obj216) else np.asarray(obj216)
        print(name216, arr216.shape,
              arr216.tolist() if arr216.size <= 16
              else {"column_norms": np.linalg.norm(arr216, axis=0).tolist(),
                    "range": [float(arr216.min()), float(arr216.max())]})
for key216, val216 in r214_results.items():
    print("contrast", key216, type(val216).__name__,
          [(type(x).__name__, tuple(x.shape)) if hasattr(x, "shape")
           else type(x).__name__ for x in val216]
          if isinstance(val216, tuple)
          else list(val216.keys()) if isinstance(val216, dict)
          else tuple(val216.shape) if hasattr(val216, "shape") else "")

print("\n=== EARLY RECORD-IDENTIFICATION NOTES ===")
print(Path("/work/notes.md").read_text()[:7000])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 216; forwards 14. Architecture confirms all drug effects traverse latent/prelude/recurrence; recovering retained replay computations and record-derived physiological targets for a symmetric-impulse test. Current candidate206 remains unchanged.\n")
```