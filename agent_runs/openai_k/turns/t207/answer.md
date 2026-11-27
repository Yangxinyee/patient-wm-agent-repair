The recovered evidence consistently supports an insulin effect near −2 glucose units per rate unit and a norepinephrine effect near +3 MAP units, with different persistence. Previous official rollouts remained much weaker. The recovery output was truncated before the useful interfaces, so I’ll retrieve only those now; this should reveal where the physical dose contrast is being lost.

```python
import inspect, re
from pathlib import Path

print("=== EXACT HELPER SIGNATURES AND DEPENDENCIES ===")
wanted191 = ["posterior_token154", "local_token154", "token164", "build_candidate89"]
local191 = [(k,v) for k,v in list(globals().items())
            if inspect.isfunction(v) and
            (re.search(r"local|replay|head|action", k, re.I))]
for k,v in local191[-14:] + [(k,globals()[k]) for k in wanted191 if k in globals()]:
    print(k, "args:", list(inspect.signature(v).parameters),
          "uses:", list(v.__code__.co_names))

print("\n=== RECENT ROUTING ARRAYS ===")
for k,v in list(globals().items()):
    if re.match(r"r18[67]_", k):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "keys", list(v)[:10])
        elif isinstance(v, (list,tuple)):
            print(k, "length", len(v))

print("\n=== CURRENT ROUTING PARAMETERS ===")
print("scales139:", globals().get("scales139"))
for k in ["r180_effects", "factual_missing_effect0"]:
    print(k, globals().get(k))
if "best188" in globals():
    print("best188 scalar diagnostics:",
          {k:v for k,v in best188.items() if k != "dirs"})
    print("best188 dirs:", type(best188["dirs"]).__name__,
          getattr(best188["dirs"], "shape", None))
for k,v in candidate139.items():
    a = torch.as_tensor(v)
    b = torch.as_tensor(load_original(k))
    print(k, tuple(a.shape), "edit norm", float((a-b).norm()))

print("\n=== COMPACT TOKEN HELPER BYTECODE ===")
for k in ["posterior_token154", "local_token154", "token164"]:
    fn = globals().get(k)
    if fn is None:
        continue
    print("\n"+k, "locals:", fn.__code__.co_varnames)
    for ins in list(dis.get_instructions(fn)):
        if ins.opname not in {"RESUME","CACHE","NOP","PRECALL","PUSH_NULL"}:
            print(ins.opname, ins.argrepr[:100])

with Path("/work/notes.md").open("a") as f:
    f.write("\n\n## Run 191: compact interface recovery\n"
            "- Run 190 output truncated; record evidence still supports insulin about -2 and NE about +3 per rate unit.\n"
            "- Retrieving only routing helper dependencies and recent fit arrays to avoid another output overflow.\n"
            "- No weights changed. Python runs: 191; forward passes: 10.\n")
```