The replay exposes both the pre-coda state and the dose-sensitive prelude features. That gives a concrete place for a low-rank edit. Before fitting it, I’ll verify the correction’s scaling and measure the local dose response. I also need to constrain non-drug bundles: an edit quiet only on drug bundles could still damage factual predictions elsewhere.

```python
import dis, io, numpy as np, torch

print("=== RECURRENT CORRECTION SEMANTICS ===")
buf242 = io.StringIO()
for ins242 in dis.get_instructions(core228):
    if ins242.offset >= 880 and ins242.opname not in {
        "RESUME", "CACHE", "NOP", "PRECALL", "PUSH_NULL", "EXTENDED_ARG"
    }:
        print(f"{ins242.offset}: {ins242.opname} {ins242.argrepr}", file=buf242)
print(buf242.getvalue()[:3300])

print("\n=== GENERAL REPLAY DEPENDENCIES ===")
print("r59_core:", r59_core.__code__.co_names)
print("repaired_core_local:", repaired_core_local.__code__.co_names)
print(compact_bytecode241(r59_core, 2100))
print("\ncontext151:", {
    k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
    for k, v in context151.items()
})

print("\n=== LOCAL DOSE READOUTS ===")
pbase242 = r49_readouts(base228["out"], base228["h"], indices151)
print("baseline return:", type(pbase242).__name__,
      tuple(pbase242.shape) if hasattr(pbase242, "shape") else "")
def np242(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("baseline means:", np242(pbase242).mean(axis=0))
for drug242, cf242 in original_cf231.items():
    pc242 = r49_readouts(cf242["out"], cf242["h"], indices151)
    print(drug242, "local doubled-dose shift:",
          (np242(pc242) - np242(pbase242)).mean(axis=0))

print("\n=== CAPTURED HEAD CONTENTS ===")
for event242, data242 in r235_result.items():
    print(event242, {
        k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        for k, v in data242.items()
    } if isinstance(data242, dict) else type(data242).__name__)

print("\n=== AVAILABLE GENERAL REPLAY ARRAYS ===")
for name242 in r59_core.__code__.co_names:
    obj242 = globals().get(name242)
    if hasattr(obj242, "shape"):
        print(name242, tuple(obj242.shape))
    elif isinstance(obj242, dict):
        print(name242, "dict keys", list(obj242)[:12])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 242: establish recurrent correction semantics\n"
            "- Original replay supplies pre-coda state and physical dose contrasts.\n"
            "- Checking correction scaling and baseline local dose sensitivities.\n"
            "- Require factual-quiet features on all bundle types, not only drug bundles.\n"
            "- No weights changed. Python runs: 242; forward passes: 17.\n")
```