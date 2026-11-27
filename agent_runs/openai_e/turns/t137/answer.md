The recovered estimates put norepinephrine’s working targets near +4.2 to +4.6 for doubling and −2.1 to −2.3 for halving. The current correction is much smaller. I’ll check whether fitting the edit to MAP-output changes, rather than a single dose probe, can reduce factual leakage. Better local selectivity would justify a GPU trial; unchanged leakage would refute this route.

```python
print("=== LOCAL OUTPUT HELPERS ===")
_budget118 = 4700
for _fnname118 in ("next_means45", "coda46", "core_update_local"):
    _fn118 = globals().get(_fnname118)
    if _fn118 is None:
        continue
    try:
        _text118 = inspect.getsource(_fn118)
    except Exception as _err118:
        _text118 = (f"{_fnname118}: source unavailable; "
                    f"args={_fn118.__code__.co_varnames[:_fn118.__code__.co_argcount]}; "
                    f"globals={_fn118.__code__.co_names}; "
                    f"constants={str(_fn118.__code__.co_consts)[:550]}")
    print(_text118[:_budget118])
    _budget118 -= min(len(_text118), _budget118)
    if _budget118 <= 0:
        break

print("\n=== RELEVANT ARRAY AND DICTIONARY INVENTORY ===")
_lines118 = []
for _name118, _obj118 in sorted(list(globals().items())):
    if _name118.startswith("_"):
        continue
    if (re.search(r"(77|95|103|107|108|114|115)$", _name118)
        or any(s in _name118.lower() for s in ("direction", "hnew", "jac", "sens", "e_fact", "basis", "memory_data"))):
        if isinstance(_obj118, (np.ndarray, torch.Tensor)):
            _lines118.append(f"{_name118}: shape={tuple(_obj118.shape)} dtype={_obj118.dtype}")
        elif isinstance(_obj118, dict):
            _lines118.append(f"{_name118}: keys={list(_obj118)[:8]}")
        elif isinstance(_obj118, (list, tuple)):
            _lines118.append(f"{_name118}: {type(_obj118).__name__} length={len(_obj118)}")
        elif isinstance(_obj118, (int, float, str, bool, np.number)):
            _lines118.append(f"{_name118}: {str(_obj118)[:90]}")
print("\n".join(_lines118)[:3000])
print("\n=== REMAINING RECENT FUNCTIONS ===")
print("\n".join(n + str(o.__code__.co_varnames[:o.__code__.co_argcount])
    for n, o in list(globals().items())
    if not n.startswith("_") and callable(o) and hasattr(o, "__code__")
    and re.search(r"(9[0-9]|10[0-9]|11[0-5])$", n))[:650])
with open("/work/notes.md", "a") as _f118:
    _f118.write("\n## Run 118: output-constrained edit feasibility\n"
                "Recovered NE schedule target +4.2..4.6 / -2.1..2.3. Scalar probe edits trade causal gain for factual leakage. "
                "Inspecting local coda/readout helpers and retained states for a closed-form output-constrained, multi-direction edit. "
                "No weights changed. Python runs: 118; forwards: 15.\n")
```