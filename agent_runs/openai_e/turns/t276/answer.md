The rollout retains much of the insulin correction (−11.85/+4.00), but norepinephrine remains far below the record-based target (+0.12/−0.88). I’ll inspect whether the prelude’s wider nonlinear features can isolate dose changes more selectively than `b_in`. Lower factual activation for comparable dose sensitivity would support that route; otherwise I’ll retain the validated joint trial.

```python
import inspect

print("PRELUDE TENSORS")
for name237 in NAMES:
    if "core.prelude" in name237:
        print(name237, SHAPES[name237])

print("\nLOCAL FORWARD / FEATURE HELPERS")
helpers237 = []
for name237, value237 in list(globals().items()):
    if inspect.isfunction(value237) and re.search(
            r"prelude|core|atom|encod|action|bundle|feature|context", name237, re.I):
        helpers237.append((name237, value237))
for name237, function237 in sorted(helpers237)[:48]:
    try:
        signature237 = str(inspect.signature(function237))
    except Exception:
        signature237 = "signature unavailable"
    print(name237, signature237[:210])

print("\nMODULES AND WEIGHT DICTIONARIES")
for name237, value237 in list(globals().items()):
    if isinstance(value237, torch.nn.Module):
        print(name237, type(value237).__name__)
    elif isinstance(value237, dict) and any(
            isinstance(k, str) and k.startswith("core.prelude")
            for k in list(value237.keys())[:200]):
        print(name237, "weight dictionary", len(value237))

print("\nRECOVERABLE FEATURE ARRAYS")
featurevars237 = [(k, v) for k, v in list(globals().items())
                  if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame))
                  and re.search(r"prelude|factual.*E|slot.*feature|Etrain|Efact|core.*input|batch|ctx|context", k, re.I)]
for name237, value237 in featurevars237[:30]:
    print(name237, type(value237).__name__, tuple(value237.shape))

print("\nEARLIER LOCAL-CORE CONTRACT NOTES")
budget237 = 1500
for section237 in sections233:
    head237 = section237.splitlines()[0] if section237.splitlines() else ""
    if re.search(r"\bRun\s+(?:39|40|41|42|43|44|45|46|47|113|114|115|116)\b", head237):
        if budget237 > 0:
            text237 = section237[:min(300, budget237)]
            print(text237)
            budget237 -= len(text237)

with Path("/work/notes.md").open("a") as f237:
    f237.write(
        "\n## Run 237: joint trial validated; inspect more selective route\n"
        "Forward28 joint236 factual cost .00441 on the challenging fresh sample. "
        "Forward29 pooled shifts insulin -11.847/+4.000, NE +.115/-.882: "
        "causal repair remains incomplete, especially NE doubling. Joint236 retained. "
        "Inspecting prelude FFN feature access for a wider, nonlinear, closed-form "
        "dose-contrast detector with factual guards. No additional weights changed. "
        "Python runs: 237; forwards: 29.\n"
    )
```