The architecture confirms that dose information must pass through the latent token, recurrent state, and shared readout. The current edit’s weak insulin response could therefore be a routing failure rather than evidence of a small drug effect. I’ll recover the record estimates and exact candidate construction so I can distinguish those possibilities without disturbing the saved weights.

```python
import inspect, dis, re
from pathlib import Path

notes190 = Path("/work/notes.md").read_text()
print("=== RECORD IDENTIFICATION AND EARLIER VALIDATION ===")
sections190 = re.split(r"\n(?=#{1,3} )", notes190)
chosen190 = [s for s in sections190 if re.search(
    r"identif|causal effect|true effect|persistence|impulse|forward.?10|candidate139|Run 139|Run 140|Run 141",
    s, re.I)]
budget190 = 15000
for s in chosen190:
    if budget190 <= 0:
        break
    excerpt = s[:min(2400, budget190)]
    print(excerpt)
    budget190 -= len(excerpt)

print("\n=== CURRENT CANDIDATE CONSTRUCTION OBJECTS ===")
for k in ["scales139", "r180_effects", "factual_missing_effect0",
          "dose_feature_response_summary", "best188"]:
    if k in globals():
        v = globals()[k]
        if isinstance(v, dict):
            for a, b in v.items():
                if isinstance(b, (np.ndarray, torch.Tensor)):
                    print(k, a, "shape", tuple(b.shape),
                          "range", float(b.min()), float(b.max()))
                else:
                    print(k, a, repr(b)[:1400])
        else:
            print(k, repr(v)[:2000])

print("\n=== LOCAL REPLAY AND ROUTING INTERFACES ===")
funcs190 = []
for k, v in list(globals().items()):
    if inspect.isfunction(v) and re.search(
        r"local|action|token|build_candidate|head|mean|187|188", k, re.I):
        funcs190.append((k, v))
for k, v in funcs190[-35:]:
    print(k, "args", list(inspect.signature(v).parameters),
          "globals", list(v.__code__.co_names),
          "locals", list(v.__code__.co_varnames)[:25])

print("\n=== RECENT FIT ARRAYS ===")
for k, v in list(globals().items()):
    if re.match(r"r18[67]_", k):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict keys", list(v)[:15])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "length", len(v))
        elif isinstance(v, (float, int, str)):
            print(k, repr(v)[:200])

print("\n=== KEY HELPER IMPLEMENTATIONS ===")
for k in ["posterior_token154", "local_token154", "token164", "build_candidate89"]:
    if k in globals():
        print("\n", k)
        try:
            print(inspect.getsource(globals()[k])[:6500])
        except (OSError, TypeError):
            dis.dis(globals()[k], depth=0)

with Path("/work/notes.md").open("a") as f:
    f.write("\n\n## Run 190: recover evidence and exact routing construction\n"
            "- Architecture places dose effects behind nonlinear latent/state routing; weak local insulin response is not evidence of a weak true effect.\n"
            "- Recovering identification estimates, candidate construction, and exact replay helpers.\n"
            "- No weights changed. Python runs: 190; forward passes: 10.\n")
```