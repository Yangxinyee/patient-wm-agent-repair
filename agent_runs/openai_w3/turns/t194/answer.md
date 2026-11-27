The saved objects include a factual-neutral attention head, a dose-innovation posterior, and recurrent replay helpers. The failed calibration appears restricted to only 48 output directions; that restriction may be unnecessarily weakening the repair. I will check the reserved head’s layout and readout calculation. If its unused value channels can carry arbitrary directions, we can test a better-conditioned correction.

```python
print("=== RECENT TRANSPORT OBJECTS ===")
for k in sorted(list(globals())):
    if any(k.endswith(str(j)) for j in range(164, 172)) and not k.startswith("_"):
        v = globals()[k]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:12])
        elif callable(v):
            print(k, "function", inspect.signature(v))
print("\n=== FUNCTION DEPENDENCIES ===")
for k in ["prelude155", "prelude_parts145", "view171", "phys171", "readout134", "posterior153"]:
    fn174 = globals().get(k)
    if callable(fn174) and hasattr(fn174, "__code__"):
        print(k, "locals", fn174.__code__.co_varnames)
        print("globals", fn174.__code__.co_names)
        print("constants", [x for x in fn174.__code__.co_consts if isinstance(x,(str,int,float))])
print("\n=== RESERVED HEAD ===")
kv174 = zero169["core.prelude.0.attn.kv.weight"]
ow174 = zero169["core.prelude.0.attn.out.weight"]
for head174 in range(4):
    print(head174,
          "key", float(kv174[head174*48:(head174+1)*48].norm()),
          "value", float(kv174[192+head174*48:192+(head174+1)*48].norm()),
          "output", float(ow174[:,head174*48:(head174+1)*48].norm()))
print("J171", J171)
print("Targets", T171)
print("Carrier/decoder", np.asarray(carrier154).shape, np.asarray(decoder170).shape)
print("Candidate posterior", [(k, tuple(v.shape)) for k,v in post154.items()])
for k in ["zin154", "zb154", "tokens145", "tokens154", "tokens164", "tokens169", "step155", "step156", "step162", "step169", "step", "core_step"]:
    if k in globals():
        v = globals()[k]
        if isinstance(v,(torch.Tensor,np.ndarray)):
            print(k, tuple(v.shape))
        elif callable(v):
            print(k, inspect.signature(v), v.__code__.co_names)
        else:
            print(k, type(v).__name__)
with open("/work/notes.md","a") as f:
    f.write("\nPython run 174; forwards 10. Inspecting reserved attention-head value/output layout and calibration restrictions. Hypothesis: changing the reserved output block can avoid the ill-conditioned original 48-direction span. No new weights installed.\n")
```