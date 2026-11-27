The records support a negative immediate insulin effect and a positive immediate norepinephrine effect, under the history-adjustment assumptions used so far. Those coefficients are not themselves the sustained rollout targets. I’ll recover the estimated lag kernels and the exact action encoder so the next test measures what the edited channel actually receives.

```python
import inspect
import dis
import io
import numpy as np
import torch

def brief146(v):
    if isinstance(v, dict):
        return {str(k): brief146(x) for k, x in list(v.items())[:12]}
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        return np.round(a, 5).tolist() if a.size <= 16 else {"shape": a.shape}
    if isinstance(v, (list, tuple)):
        return [brief146(x) for x in v[:12]]
    if np.isscalar(v):
        return v
    return type(v).__name__

print("=== RECORD-DERIVED DYNAMIC TARGETS ===")
for name146 in ["kernels137", "readout_estimates137", "quadratic137"]:
    print(name146, brief146(globals().get(name146)))

print("\n=== CONTROLLED ACTION FIXTURES ===")
print("controlled_action_repair:", brief146(controlled_action_repair))
print("U65:", type(U65).__name__, getattr(U65, "shape", None))
print("action_fixture columns:", list(action_fixture_repair.columns))
print(action_fixture_repair.head(5).to_string(index=False))

print("\n=== FUNCTION IMPLEMENTATIONS ===")
budget146 = 4800
for name146 in ["rp_action_encoder", "singleton_action_repair", "atom_exact65"]:
    fn146 = globals()[name146]
    try:
        text146 = inspect.getsource(fn146)
    except (OSError, TypeError):
        stream146 = io.StringIO()
        dis.dis(fn146, file=stream146)
        text146 = stream146.getvalue()
    print(name146, "locals:", fn146.__code__.co_varnames,
          "dependencies:", fn146.__code__.co_names)
    take146 = min(len(text146), 1500, budget146)
    print(text146[:take146])
    budget146 -= take146

print("\n=== TRANSITION RETURN / CAPTURE INTERFACES ===")
print("transition73 locals:", transition73.__code__.co_varnames)
print("pre_context_repair:", inspect.signature(pre_context_repair),
      pre_context_repair.__code__.co_varnames)
print("capture candidates:", [
    k for k, v in list(globals().items())
    if callable(v) and hasattr(v, "__code__")
    and any(s in k.lower() for s in ["capture", "prelude", "replay"])
][:25])

with open("/work/notes.md", "a") as f146:
    f146.write(
        "\n## Run 146\n"
        "Recovered lag-kernel/readout estimates and action encoder interfaces. "
        "Immediate dose coefficients are not equated with sustained intervention effects. "
        "No weights changed.\n"
    )
```