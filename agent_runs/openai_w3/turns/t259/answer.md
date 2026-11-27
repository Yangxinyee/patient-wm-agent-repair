The records support substantial insulin and norepinephrine effects, unlike the original model’s nearly null rollouts. The current repair has recovered some effect but not the required symmetry. I will inspect where its signal is inserted and how factual compensation is computed; a correction that cancels itself on generated histories would explain the mismatch.

```python
import dis

print("EARLY IDENTIFICATION NOTES")
for s in pathlib.Path("/work/notes.md").read_text().splitlines():
    if re.search(r"Python runs?: (?:[1-9]|1[0-9]);|Python runs: 6;", s):
        print(s[:1500])

print("\nIMPORTANT RETAINED ARRAYS")
r231_names = [n for n in globals()
              if re.search(r"(169|186|191|192|199|206|220)$", n)]
for n in r231_names[:65]:
    v = globals()[n]
    if hasattr(v, "shape"):
        print(n, tuple(v.shape))
    elif isinstance(v, (float, int, str)):
        print(n, repr(v)[:100])
    elif isinstance(v, dict):
        print(n, "keys", list(v)[:12])

print("\nCOMPENSATION CANDIDATE")
for k, v in refit220[1.0].items():
    print(k, tuple(v.shape) if hasattr(v, "shape") else repr(v)[:200])

print("\nEXACT INSERTION PATH")
for name, limit in [("prelude155", 3300), ("view171", 1300),
                    ("phys171", 900), ("readout134", 600)]:
    f = globals()[name]
    print("\n", name, "locals:", f.__code__.co_varnames)
    print(" | ".join(f"{i.opname} {i.argrepr}" for i in dis.Bytecode(f))[:limit])

print("\nDIRECTION OBJECT NAMES")
print([n for n in globals() if re.search(r"dir|rho|causal|beta", n, re.I)][:75])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 231; forwards 16. Recovering the exact FFN insertion and compensation decomposition. Working hypothesis: nonlinear/off-manifold transport or generated-history-dependent compensation causes dose asymmetry; identification remains based only on records. No weights changed.\n")
```