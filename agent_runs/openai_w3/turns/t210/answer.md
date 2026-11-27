The earlier record-based targets are about −13 glucose units for doubled insulin and +4.3 MAP units for doubled norepinephrine, with opposite half-dose effects. These are estimates, not observed counterfactual truth. I’ll inspect the retained transport and response functions before installing a pilot. It should preserve factual fit while moving rollouts toward those estimates.

```python
import dis, linecache
print("=== TRANSPORT / RESPONSE HELPERS ===")
for name in ["latent187", "response186", "preludeE185", "prelude155",
             "posterior153", "view171", "phys171"]:
    fun = globals()[name]
    print("\n", name, "globals:", fun.__code__.co_names,
          "constants:", repr(fun.__code__.co_consts)[:250])
    try:
        src = inspect.getsource(fun)
    except Exception:
        src = "".join(linecache.getlines(fun.__code__.co_filename))
        if not src:
            src = dis.Bytecode(fun).dis()
    print(src[:1500])

print("\n=== DEPENDENCY SHAPES ===")
deps190 = set()
for name in ["latent187", "response186", "preludeE185", "prelude155"]:
    deps190.update(globals()[name].__code__.co_names)
for name in sorted(deps190):
    obj = globals().get(name)
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape), obj.dtype)
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), list(obj)[:8])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))
for key, val in decoders187.items():
    print("decoder", key, type(val).__name__)
    if isinstance(val, (list, tuple)):
        print([(type(v).__name__, getattr(v, "shape", None)) for v in val])
    elif isinstance(val, dict):
        print({k: getattr(v, "shape", type(v).__name__) for k, v in val.items()})
    else:
        print(getattr(val, "shape", None))
print("post187 differences:",
      {k: float(np.linalg.norm(np179(v)-np179(load_original(k))))
       for k, v in post187.items()})
print("Saved neutral checkpoint:", list(zero169))
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 190; forwards 10. Recovered record-derived reference means: insulin x2 -12.9555, half +6.4778; NE x2 +4.2797, half -2.1398 on first 100 tuning stays. Inspecting exact latent carrier and unrestricted prelude-response helpers before a reversible actual-rollout pilot. No weight changes.\n")
```
