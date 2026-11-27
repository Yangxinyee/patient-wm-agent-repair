The records place doses after the same-hour readings, and the earlier history-adjusted analyses estimated next-hour insulin effects near −2 and norepinephrine effects near +3 per rate unit. These are observational estimates, so I’ll recover their dynamic checks and the retained editing helpers. That will show whether the weak rollout response reflects a routing problem rather than a weak drug effect.

```python
print("=== RETAINED NAMES ===")
_names201 = sorted(k for k in globals()
                   if not k.startswith("_") and k not in {"In", "Out", "get_ipython"})
print(" ".join(_names201)[:5200])
print("\n=== EARLY DYNAMICS AND TARGETS ===")
print(Path("/work/notes.md").read_text()[6500:9300])
print("\n=== FORWARD 13 FACTUAL CHECK ===")
for _p201 in sorted(Path("/work/fwd_13").glob("*.csv")):
    _df201 = pd.read_csv(_p201)
    print(_p201.name, _df201.shape, list(_df201.columns))
    for _c201 in _df201.columns:
        if "nll" in _c201.lower():
            print(_c201, float(_df201[_c201].mean()))
print("\n=== CURRENT EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 201: recover dynamics and current factual margin\n"
            "- Architecture and schema recovered. Earlier cross-fitted immediate "
            "effects were approximately insulin -2 and NE +3 per rate unit.\n"
            "- Retrieving retained interfaces and official forward-13 factual "
            "loss before testing drug-specific directions. No weights changed.\n"
            "- Python runs: 201; forward passes: 14.\n")
```