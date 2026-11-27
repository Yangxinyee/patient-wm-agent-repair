The original rollout responses are much smaller than the record-based estimates—for example, insulin doubling shifts glucose by less than one unit on this sample. The corrected routine also uses common latent draws across branches, which is essential for comparing small state changes. I’ll inspect the latent update and decoder fits next; a viable edit needs a dose-innovation signal that generalizes across stays.

```python
print("=== CACHE AND DECODER INVENTORY ===")
wanted = ("cache", "decoder", "decode", "ridge", "innovation", "latent",
          "target", "coef", "features", "fit", "weights", "valid", "panel")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or callable(v) or not any(t in k.lower() for t in wanted):
        continue
    if isinstance(v, pd.DataFrame):
        print(k, "DF", v.shape, list(v.columns)[:16])
    elif isinstance(v, dict):
        print(k, "DICT", {
            str(a): tuple(b.shape) if isinstance(b, (np.ndarray, torch.Tensor))
            else ("DF" + str(b.shape) if isinstance(b, pd.DataFrame)
                  else repr(b)[:80] if isinstance(b, (str, int, float, tuple))
                  else type(b).__name__)
            for a, b in list(v.items())[:16]})
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, tuple(v.shape))
    elif isinstance(v, (float, int, str, tuple)):
        print(k, repr(v)[:160])

def compact_impl79(fn):
    print("\nFUNCTION", fn.__name__, inspect.signature(fn))
    print("CONSTANTS:", repr(fn.__code__.co_consts)[:1300])
    grouped = defaultdict(list)
    line = None
    abbreviations = {
        "LOAD_GLOBAL": "G", "LOAD_FAST": "L", "STORE_FAST": "S",
        "LOAD_CONST": "C", "LOAD_ATTR": "A", "BINARY_SUBSCR": "[]",
        "BINARY_OP": "OP", "CALL": "call", "KW_NAMES": "kw",
        "RETURN_VALUE": "return", "STORE_SUBSCR": "[]= "}
    for ins in dis.Bytecode(fn):
        if ins.starts_line is not None:
            line = ins.starts_line
        if ins.opname in ("RESUME", "CACHE", "PRECALL", "PUSH_NULL"):
            continue
        arg = ins.argrepr.replace("NULL + ", "").replace("NULL|self + ", "")
        grouped[line].append(abbreviations.get(ins.opname, ins.opname) + ":" + arg)
    for line, parts in grouped.items():
        print(line, " ".join(parts))

for name in ("latent_step66", "local_step67", "posterior_features4"):
    compact_impl79(globals()[name])
```