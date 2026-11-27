Factual fit passes, but the NE rollouts have the wrong signs: doubling lowers MAP and halving raises it. That refutes the immediate-response calibration as a guide to sustained rollouts. I’ll inspect the recurrent-response construction and matched record targets before changing weights again. A useful revision must correct those signs while retaining factual fit.

```python
g200 = dict(globals())
backup_passing200 = {
    n: (load(n).clone() if hasattr(load(n), "clone") else np.array(load(n), copy=True))
    for n in edited()
}
d200 = pd.read_csv("/work/fwd_17/dose_edit_edited.csv")
d200 = d200[d200["reached"].eq(True)]
print("CURRENT ROLLOUT, ROW MEANS", d200.groupby("edit")["shift"].mean().to_dict())
print("CURRENT ROLLOUT, EQUAL-STAY MEANS",
      d200.groupby(["stay_id", "edit"])["shift"].mean().groupby("edit").mean().to_dict())

print("RESPONSE-DESIGN HISTORY")
for block200 in blocks197:
    match200 = re.match(r"## Run (\d+)", block200)
    if match200 and 138 <= int(match200.group(1)) <= 170:
        print(block200[:210].replace("\n", " "))
print("RESPONSE AND TARGET OBJECTS")
print(" ".join(n for n in sorted(g200)
               if re.search(r"137|148|153|155|159|160|163|165|166|167", n))[:2400])
for n200 in ["J187", "directions_ne187", "ins_direction187",
             "reference_bank125", "reference_x125", "repair_bank", "submitted193"]:
    if n200 in g200:
        x200 = g200[n200]
        if isinstance(x200, dict):
            print(n200, "keys", list(x200)[:16])
        else:
            print(n200, describe198(x200))
for n200 in ["transition73", "replay_factual125", "capture_transition102"]:
    f200 = g200.get(n200)
    if inspect.isfunction(f200):
        print(n200, inspect.signature(f200), "GLOBALS", f200.__code__.co_names,
              "LOCALS", f200.__code__.co_varnames)
print("WEIGHT TYPES", type(next(iter(rW.values()))).__name__,
      type(next(iter(submitted193.values()))).__name__)
with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 200\nScale-.5 NE edit passed GPU factual (97 tuning stays, dNLL +.00233) but failed NE rollout signs (row means x2 -1.131, half +.797). Insulin row means x2 -10.463, half +12.317. Saved passing factual candidate in backup_passing200. Immediate NE Jacobian calibration is not sufficient for sustained rollout repair; investigating recurrent response/decoder feedback. Tool summaries explicitly report stay-hour means despite architecture's stay-averaging description, so retain both aggregates and matched targets. No weights changed.\n")
```