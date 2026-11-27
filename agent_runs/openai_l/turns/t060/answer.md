The architecture confirms that doses affect the recurrent state through both the action encoder and posterior. The earlier notes also contain record-based evidence of effects larger than the original rollouts. I’ll recover the relevant functions and diagnostics without another oversized dump. If the remaining reconstruction error is confined to measurement bundles, we can test action edits without rebuilding every component.

```python
import inspect, types
print("=== CAUSAL EVIDENCE, CONTINUED ===")
print(Path("/work/notes.md").read_text()[5000:10000])

print("\n=== LOCAL FUNCTIONS ===")
fn53 = {}
for nm53, ob53 in list(globals().items()):
    if isinstance(ob53, types.FunctionType) and ob53.__module__ == "__main__":
        fn53[nm53] = ob53
for nm53 in sorted(fn53):
    print(nm53, str(inspect.signature(fn53[nm53])))

print("\n=== RECENT DIAGNOSTICS AND CORE DATA ===")
for nm53 in sorted(list(globals())):
    if any(s in nm53.lower() for s in [
        "49", "50", "51", "52", "kernels19", "r32_best",
        "state_train_ids36", "d22", "bundle_map", "posterior"
    ]):
        ob53 = globals()[nm53]
        if isinstance(ob53, types.FunctionType):
            continue
        desc53 = str(getattr(ob53, "shape", ""))
        if isinstance(ob53, dict):
            desc53 += " keys=" + repr(list(ob53)[:12])
        elif isinstance(ob53, (list, tuple)):
            desc53 += " len=" + str(len(ob53))
            if len(ob53) <= 4:
                desc53 += " value=" + repr(ob53)[:400]
        elif isinstance(ob53, (str, int, float)):
            desc53 += " value=" + repr(ob53)[:200]
        print(nm53, type(ob53).__name__, desc53)

print("\n=== RECENT FUNCTION REFERENCES ===")
for nm53 in sorted(fn53):
    if any(s in nm53 for s in ["45", "49", "50", "51", "52", "post", "encode", "atom"]):
        ob53 = fn53[nm53]
        print(nm53, "vars=", ob53.__code__.co_varnames,
              "refs=", ob53.__code__.co_names)
```
