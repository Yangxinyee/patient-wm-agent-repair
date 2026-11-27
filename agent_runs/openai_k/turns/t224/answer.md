The retained objects include both the record-based dynamics and the six-tensor reserved-channel construction. The last inspection overflowed while printing a function’s default tensors, so it didn’t reveal the head implementation. I’ll inspect it without printing defaults; agreement with official factual predictions is the prerequisite for trusting a new calibration.

```python
def args203(fn):
    c = fn.__code__
    return c.co_varnames[:c.co_argcount + c.co_kwonlyargcount]

print("=== HEAD AND TOKEN FUNCTIONS ===")
for _k203, _x203 in list(globals().items()):
    if isinstance(_x203, types.FunctionType) and _x203.__module__ == "__main__":
        if any(s in _k203.lower() for s in
               ["readout", "head", "token", "local", "surrog", "predictions"]):
            print(_k203, args203(_x203),
                  "deps:", " ".join(_x203.__code__.co_names)[:230])

print("\n=== CANDIDATE AND CALIBRATION OBJECT NAMES ===")
_sel203 = [k for k in sorted(globals()) if
           any(s in k.lower() for s in
               ["zero", "139", "138", "188", "surrog", "target", "readout", "151"])]
print(" ".join(_sel203)[:1700])

print("\n=== DYNAMIC EFFECTS ===")
print(dynamic_ir)
print("dose channel assay:", dose_feature_response_summary)

print("\n=== RELEVANT NOTEBOOK SECTIONS ===")
_n203 = Path("/work/notes.md").read_text()
for _heading203 in ["## Run 6:", "## Run 7:", "## Run 104:",
                    "## Run 110:", "## Run 118:", "## Run 119:"]:
    _a203 = _n203.find(_heading203)
    if _a203 >= 0:
        _b203 = _n203.find("\n## ", _a203 + 4)
        print(_n203[_a203:_b203 if _b203 >= 0 else None][:700])

print("\n=== HELPER SOURCE AVAILABILITY ===")
for _k203 in ["r49_readouts", "local_token154", "posterior_token154", "token164"]:
    if _k203 in globals():
        _fn203 = globals()[_k203]
        try:
            _src203 = inspect.getsource(_fn203)
            print(_k203, _src203[:1400])
        except Exception as _ex203:
            print(_k203, "source unavailable:", type(_ex203).__name__,
                  "filename:", _fn203.__code__.co_filename)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 203: bounded head-source recovery\n"
            "- Run 202 confirmed retained physical channels and six-tensor "
            "candidate interfaces; output overflow hid the head details.\n"
            "- Avoiding tensor-valued function defaults and retrieving focused "
            "head/token dependencies. No weights changed.\n"
            "- Python runs: 203; forward passes: 14.\n")
```