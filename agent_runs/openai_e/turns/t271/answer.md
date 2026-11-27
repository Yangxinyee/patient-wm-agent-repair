The recovered records show why simple dose associations were unreliable: adjustment and threshold analyses reversed them, provisionally supporting negative insulin and positive norepinephrine effects. I’ll now retrieve the later estimates and tested edit trade-offs. Consistent later estimates would justify calibration toward those targets; disagreement would call for more caution.

```python
print("LATER CALIBRATION SUMMARIES")
wanted234 = {159, 160, 163, 166, 169, 177, 178, 185, 186,
             189, 199, 205, 208, 211, 213, 216, 219, 220, 221}
budget234 = 6100
for sec234 in sections233:
    heading234 = sec234.splitlines()[0] if sec234.splitlines() else ""
    nums234 = {int(x) for x in re.findall(r"\b(?:Run|Runs)\s+(\d+)", heading234)}
    if nums234 & wanted234 and budget234 > 0:
        text234 = sec234[:min(430, budget234)]
        print(text234)
        budget234 -= len(text234)

print("\nARCHIVE KEYS")
for file234 in ["baseline77.npz", "insulin_trial219.npz",
                "ne_candidate151.npz", "ne_candidate199.npz"]:
    with np.load("/work/" + file234, allow_pickle=False) as a234:
        print(file234, {k: a234[k].shape for k in a234.files})

print("\nTARGET / IMPULSE VARIABLES")
vars234 = [(k, v) for k, v in list(globals().items())
           if not k.startswith("_") and
           re.search(r"beta|impulse|target|frontier", k, re.I)]
for k234, v234 in vars234[:35]:
    if isinstance(v234, (np.ndarray, torch.Tensor)):
        print(k234, type(v234).__name__, tuple(v234.shape))
    elif isinstance(v234, (int, float, str)):
        print(k234, repr(v234)[:140])
    elif isinstance(v234, dict):
        print(k234, "dict keys:", list(v234)[:12])
    elif isinstance(v234, pd.DataFrame):
        print(k234, "DataFrame", v234.shape, list(v234.columns)[:10])

with Path("/work/notes.md").open("a") as f234:
    f234.write(
        "\n## Run 234: bounded calibration recovery\n"
        "Recovered early identification: insulin RD/impulse estimates were negative "
        "despite positive raw association; NE history-adjusted and residual-dose "
        "estimates were positive despite negative raw association. Retrieving later "
        "targets and tested candidates with bounded output; no weights changed. "
        "Python runs: 234; forwards: 27.\n"
    )
```