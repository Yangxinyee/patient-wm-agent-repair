The recovered notes put the six-lag record-based targets near −12.37/+6.19 for insulin and +4.38/−2.19 for norepinephrine. The current edit remains too weak, especially for halved norepinephrine. The output was truncated before the factual-loss results, so I’ll extract those and the correction’s exact variables compactly. If factual loss has sufficient headroom, a stronger or separately calibrated correction may be viable.

```python
print("=== FACTUAL PASS 18 ===")
for fn in sorted(os.listdir("/work/fwd_18")):
    if not fn.endswith(".csv"):
        continue
    df18 = pd.read_csv("/work/fwd_18/" + fn)
    nc18 = [c for c in df18.columns if "nll" in c.lower()]
    print(fn, "rows", len(df18), "columns", ",".join(df18.columns))
    for c in nc18:
        if pd.api.types.is_numeric_dtype(df18[c]):
            print(c, float(df18[c].mean()))
            if "event" in df18:
                print(df18.groupby("event")[c].mean().round(6).to_dict())
for fn in sorted(os.listdir("/work/fwd_18")):
    if fn.endswith(".json"):
        j18 = json.load(open("/work/fwd_18/" + fn))
        print(fn, str(j18)[:1400])

snap253 = list(globals().items())
print("\n=== OBJECT NAMES FROM RUNS 243 ONWARD ===")
recent253 = []
for n, v in snap253:
    m = re.search(r"(\d{3})$", n)
    if m and 243 <= int(m.group(1)) <= 251:
        if isinstance(v, (np.ndarray, torch.Tensor)):
            desc = str(tuple(v.shape))
        elif isinstance(v, dict):
            desc = "keys=" + ",".join(map(str, list(v)[:10]))
        elif isinstance(v, (tuple, list)):
            desc = "length=" + str(len(v))
        elif isinstance(v, (float, int, str, np.number)):
            desc = str(v)[:80]
        else:
            desc = type(v).__name__
        recent253.append(n + ":" + desc)
print("\n".join(recent253[-85:]))

print("\n=== REPLAY FUNCTION INTERFACES ===")
for n, v in snap253:
    if inspect.isfunction(v) and re.search(r"replay|core.*step|value.*eval|head.*eval|head.*mean", n, re.I):
        print(n, list(inspect.signature(v).parameters))
print("\n=== CORRECTION AND TARGET OBJECT NAMES ===")
print(" ".join(n for n, v in snap253
               if not n.startswith("_") and re.search(r"corr|target|route|strength|ratefit", n, re.I)))
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 253: compact factual and candidate recovery\n"
        "- Recovered six-lag record targets: insulin x2 -12.3739/half +6.1870; "
        "NE x2 +4.3837/half -2.1918 (effects beyond lag six omitted).\n"
        "- Current half-strength edit remains weak; extracting forward 18 NLL and exact candidate "
        "objects without verbose architecture/notes output.\n"
        "- No weights changed. Python runs: 253; forward passes: 19.\n"
    )
```
