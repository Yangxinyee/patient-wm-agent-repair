The notebook establishes record-based targets of about −13 and +6.5 for insulin, and +4.3 and −2.1 for norepinephrine on the previously tested stays. The output cut off before the implementation inventory, so I’ll retrieve that compactly. This should let me distinguish a faulty transport signal from feedback introduced by factual compensation.

```python
import inspect
g237 = dict(globals())
fn237 = sorted(k for k,v in g237.items()
               if inspect.isfunction(v) and not k.startswith("_"))
print("FUNCTIONS\n" + " ".join(fn237))
print("\nSELECTED VARIABLES")
for k,v in sorted(g237.items()):
    if not any(s in k.lower() for s in
               ["193", "199", "206", "220", "221", "222",
                "228", "234", "235", "checkpoint", "transport"]):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict keys:", list(v)[:18])
    elif isinstance(v, (list,tuple)):
        print(k, "length:", len(v))
    elif isinstance(v, (int,float,bool)):
        print(k, v)
    elif isinstance(v,str) and len(v)<100:
        print(k, v)
print("\nINTERMEDIATE NOTEBOOK HIGHLIGHTS")
n237 = Path("/work/notes.md").read_text()
lines237 = [s for s in n237.splitlines()
            if ("Python run" in s and
                any(t in s for t in [" 100", " 110", " 120", " 130", " 140",
                                    " 150", " 160", " 170", " 180", " 190",
                                    " 193", " 199", " 200", " 206"]))]
print("\n".join(lines237)[:2200])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 237; forwards 16. Recovered record-derived main rollout references from early notebook. Retrieving compact implementation inventory after run 236 output truncation. No weights changed.\n")
```