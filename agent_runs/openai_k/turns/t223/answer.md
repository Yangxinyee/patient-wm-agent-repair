The current edit raises factual value NLL by 0.00231 nats per reading, within the 0.005 limit, but its drug responses are still inadequate. I’ll inspect the reserved-dose channels and local prediction helpers. If those helpers disagree with official predictions, I’ll treat their calibration results as unreliable.

```python
import inspect
import types

def brief202(x):
    if isinstance(x, dict):
        return {"type": "dict", "len": len(x), "keys": list(x)[:18]}
    if isinstance(x, pd.DataFrame):
        return {"type": "DataFrame", "shape": x.shape, "cols": list(x.columns)[:12]}
    if hasattr(x, "shape"):
        return {"type": type(x).__name__, "shape": tuple(x.shape)}
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "len": len(x)}
    return str(x)[:180]

print("=== SELECTED RETAINED OBJECTS ===")
for _k202 in ["candidate139", "candidate193", "candidate199", "backup199",
              "best188", "dynamic_ir", "dml_results", "core_reconstruction_results",
              "dose_feature_response_summary", "candidate158", "W0"]:
    if _k202 in globals():
        _x202 = globals()[_k202]
        print(_k202, brief202(_x202))
        if isinstance(_x202, dict):
            print({k: brief202(v) for k, v in list(_x202.items())[:10]})

print("\n=== RELEVANT FUNCTION INTERFACES ===")
for _k202, _x202 in list(globals().items()):
    if isinstance(_x202, types.FunctionType) and _x202.__module__ == "__main__":
        if any(s in _k202.lower() for s in
               ["head", "core", "post", "candidate", "surrog", "replay", "pred",
                "feature", "capture", "token", "physical", "compress", "zero"]):
            print(_k202, str(inspect.signature(_x202)),
                  "deps:", " ".join(_x202.__code__.co_names)[:200])

print("\n=== RESERVED-CHANNEL RELATED NAMES ===")
print(" ".join(k for k in sorted(globals()) if
               any(s in k.lower() for s in ["zero", "139", "138", "188", "surrog", "target"])))
print("\n=== ORIGINAL DYNAMIC EVIDENCE ===")
_n202 = Path("/work/notes.md").read_text()
_a202 = _n202.find("## Run 6:")
_b202 = _n202.find("## Run 9")
print(_n202[_a202:_b202][:2000])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 202: bounded routing-interface audit\n"
            "- Current factual delta is +0.002308 nats/reading, inside the +0.005 "
            "limit but not enough evidence to accept incorrect drug responses.\n"
            "- Inspecting reserved-channel candidates, core/head helpers, and "
            "record-derived dynamics. No weights changed.\n"
            "- Python runs: 202; forward passes: 14.\n")
```