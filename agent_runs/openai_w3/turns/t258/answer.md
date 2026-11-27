The architecture confirms that dose signals pass through recurrence, while generated readings also feed back through same-sensor registers. Either pathway could distort an initially symmetric response. I will inspect the replay helpers and record-derived dynamics; this should distinguish a faulty transport direction from feedback that cancels the intended effect.

```python
import inspect, re, dis

print("RECORD-DERIVED FINDINGS")
r230_lines = pathlib.Path("/work/notes.md").read_text().splitlines()
r230_matches = [s for s in r230_lines
                if re.search(r"causal|identif|regress|coefficient|rho|confound|instrument", s, re.I)]
for s in r230_matches[:18]:
    print(s[:230])

print("\nRECENT ARRAY/DICTIONARY INVENTORY")
for n, v in list(globals().items()):
    if re.search(r"(22[3-8]|checkpoint222|refit220|candidate206|restricted199)", n):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(n, tuple(v.shape))
        elif isinstance(v, dict):
            print(n, {k: (tuple(x.shape) if hasattr(x, "shape") else type(x).__name__)
                      for k, x in list(v.items())[:15]})

print("\nREPLAY IMPLEMENTATION")
for n in ["response_original228", "raw_hidden228", "fn223",
          "readout134", "latent206"]:
    v = globals().get(n)
    if not inspect.isfunction(v):
        continue
    print("\nFUNCTION", n)
    try:
        print(inspect.getsource(v)[:1500])
    except Exception:
        print("locals", v.__code__.co_varnames)
        print("names", v.__code__.co_names)
        print("constants", repr(v.__code__.co_consts)[:500])
        print("bytecode", " | ".join(
            f"{i.opname} {i.argrepr}" for i in dis.Bytecode(v)
        )[:1400])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 230; forwards 16. Architecture confirms recurrent action transport and generated-reading register feedback. Inspecting replay helpers and record-derived causal dynamics to isolate the source of asymmetric rollout effects. No edits changed.\n")
```