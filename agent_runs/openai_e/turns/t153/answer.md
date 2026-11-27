The inputs reveal the reshape cause: 96 query rows were paired with memory for only four rows. I’ll fix the row selection and test the attention variants against the GPU states. A small hidden-state error would support using the local model; a large error would mean another part is still wrong.

```python
import dis, numpy as np, torch

print("Memory helper metadata:")
for n132 in ["ff128", "core_update_local", "recon_ctx", "core_layer", "core_attn"]:
    f132 = globals().get(n132)
    if callable(f132) and hasattr(f132, "__code__"):
        c132 = f132.__code__
        print(n132, "args", c132.co_varnames[:c132.co_argcount],
              "locals", c132.co_varnames, "globals", c132.co_names,
              "constants", tuple(x for x in c132.co_consts
                                 if isinstance(x, (str, int, float, tuple))))
        if n132 == "ff128":
            print(" ".join(f"{i.opname}:{i.argrepr}" for i in dis.Bytecode(f132)
                           if i.opname not in {"RESUME", "CACHE", "PRECALL"}))

print("\nMemory/state source arrays:")
for n132, v132 in list(globals().items()):
    if isinstance(v132, (np.ndarray, torch.Tensor)) and (
        any(s in n132.lower() for s in
            ["memory", "mem_", "age_", "valid", "flow", "static", "bundle", "index"])
        or (v132.ndim >= 2 and v132.shape[0] == 4356)
        or n132.endswith(("120", "46"))
    ):
        print(n132, tuple(v132.shape))
print("\nUseful containers:")
for n132, v132 in list(globals().items()):
    if isinstance(v132, dict) and any(s in n132.lower() for s in
                                     ["local", "recon", "state", "data", "bundle", "ctx"]):
        print(n132, list(v132)[:18])
print("\nCoda/readout helpers:")
for n132, v132 in list(globals().items()):
    if callable(v132) and hasattr(v132, "__code__") and any(
        s in n132.lower() for s in ["coda", "h_to", "decode", "predict_mean"]
    ):
        print(n132, v132.__code__.co_varnames[:v132.__code__.co_argcount],
              v132.__code__.co_names)
print("\nCurrent attention identity:", core_attn is oldattn129, core_attn is attn129)
with open("/work/notes.md", "a") as f132:
    f132.write("\n## Run 132: identify batch-selection error\n"
               "Run131 showed query batch 96 but memory batch 4, explaining Run129's "
               "reshape failure. Recovering memory helper indexing for a correctly "
               "batched hidden-state validation. No weights changed. Python runs: "
               "132; forwards: 15.\n")
```