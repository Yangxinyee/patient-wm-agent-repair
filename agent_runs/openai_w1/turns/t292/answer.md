Earlier attempts could not safely amplify the value head, and the attention edits transmitted little dose response. I will examine the recurrent block’s additive input path, which enters after the blocks rather than before them. This route is promising only if a fitted dose contrast survives to the reading head with substantially less factual disturbance.

```python
import dis

print("=== DIRECT RECURRENT-PATH TENSORS ===")
for n in NAMES:
    if any(s in n for s in ["core.b_in", "core.c_out", "core.delta_in",
                            "core.a_in", "core.prelude_norm"]):
        x = np.asarray(load(n))
        print(n, x.shape, "RMS", float(np.sqrt(np.mean(x*x))))

print("\n=== CORE / READOUT HELPERS ===")
for name in ["transition214", "finite_readout221", "value_raw221",
             "rx_layer", "rx_layer2", "rx_linear", "rx_norm",
             "r244_attention_features", "ff"]:
    if name not in globals():
        continue
    f = globals()[name]
    print("\n", name, inspect.signature(f))
    try:
        s = inspect.getsource(f)
        print(s[:3500] if name in ["transition214", "finite_readout221"] else s[:700])
    except Exception:
        print("Referenced names:", f.__code__.co_names)
        print("Constants:", [c for c in f.__code__.co_consts
                              if isinstance(c, (str, int, float, tuple))][:25])

print("\n=== RECURRENT CACHE OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if re.search(r"(214|215|220|221)", k):
        if isinstance(v, dict):
            print(k, "dict", list(v.keys())[:16])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape, list(v.columns)[:12])
        elif not callable(v) and isinstance(v, (str, list, tuple)):
            print(k, type(v).__name__, str(v)[:160])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 256\n"
            "Earlier value-head amplification failed its factual/contrast tradeoff; prelude-attention "
            "transmission was weak. Investigating core.b_in, the additive recurrent injection after "
            "the blocks, as a lower-distortion route. No weights changed. "
            "Counts: 256 Python runs, 27 forwards.\n")
```