The edited responses accumulate over time, unlike the original model’s nearly flat responses. NE doubling initially lowers MAP before its response turns positive. I’ll recover the record-derived response curves and the edit’s construction to separate delayed drug effects from unintended feedback. A mismatch in their timing would support revising the dynamics.

```python
print("=== NOTEBOOK: RECORD IDENTIFICATION AND EDIT CONSTRUCTION ===")
notes220 = Path("/work/notes.md").read_text()
sections220 = notes220.split("\n## ")
selected220 = []
for s in sections220:
    first = s.splitlines()[0] if s.splitlines() else ""
    if any(x in first for x in [
        "Run 1", "Run 2", "Run 3", "Run 4", "Run 5",
        "Run 65", "Run 73", "Run 102", "Run 125", "Run 137",
        "Run 159", "Run 166", "Run 184", "Run 185", "Run 186",
        "Run 187", "Run 188", "Run 189", "Run 190", "Run 191",
        "Run 192", "Run 193", "Run 194", "Run 195"
    ]):
        if len(selected220) < 35:
            selected220.append(s[:1800])
print("\n## ".join(selected220)[:23000])

print("\n=== CANDIDATE/RESPONSE GLOBALS ===")
for name, obj in list(globals().items()):
    if any(x in name.lower() for x in [
        "193", "194", "195", "206", "209", "210",
        "kernel", "readout_estimates", "quadratic137", "target", "baseline205"
    ]):
        if inspect.isfunction(obj):
            print(name, inspect.signature(obj), "constants:", obj.__code__.co_consts[:10],
                  "globals:", obj.__code__.co_names)
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:18])
        elif isinstance(obj, pd.DataFrame):
            print(name, "DataFrame", obj.shape, "columns", list(obj.columns)[:20])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, (str, int, float, list, tuple)):
            print(name, repr(obj)[:400])

print("\n=== IMPULSE TABLES ===")
for drug, tab in impulse_tables_repair.items():
    print(drug)
    print(tab.to_string(index=False))
for key, val in impulse_tests166.items():
    print("impulse_tests166", key, repr(val)[:1800])

print("\n=== HEAD AND REPLAY FUNCTIONS ===")
for name in ["capture_transition102", "replay_factual125",
             "evaluate_states129", "head_features_fixed", "collect_channel186"]:
    obj = globals().get(name)
    if inspect.isfunction(obj):
        print(name, inspect.signature(obj), "constants:", obj.__code__.co_consts[:15],
              "globals:", obj.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 220\nHourly GPU profiles show accumulating insulin effects and NE doubling with an early wrong sign. "
            "Recovering record-derived dynamics and candidate construction to distinguish delayed response from feedback "
            "distortion. No saved weights changed.\n")
```
