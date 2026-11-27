The recovered tests are encouraging: across all 400 tuning stays, the edit increased factual NLL by only 0.000307 nats, well below the 0.005 limit. Its four tested rollout shifts were exactly zero.

That validates the edit’s behavior, but not the true effects. I’ll review the remaining counter-evidence—especially norepinephrine curvature and delayed insulin associations. Replicated dose responses would argue against zero; unstable results across independent stays would favor the current conservative edit.

```python
text = Path("/work/notes.md").read_text()
lo = text.find("## Python run 15")
hi = text.find("## Python run 21", lo)
if hi < 0:
    hi = text.find("## Run 21", lo)
if hi < 0:
    hi = text.find("## Python run 22", lo)

print("=== REMAINING EARLIER AUDIT NOTES ===")
for line in text[lo:hi].splitlines():
    if len(line) > 700:
        print(line[:550] + " ... [long result omitted]")
    else:
        print(line)

print("\n=== NOREPINEPHRINE REPLICATION SUMMARY ===")
def concise_results(obj, prefix=""):
    if not isinstance(obj, dict):
        return
    fields = ["n", "stays", "coefficients", "cluster_se",
              "joint_null_p", "joint_dose_p",
              "x2_conditional_shift", "half_conditional_shift"]
    found = {k: obj[k] for k in fields if k in obj}
    if found:
        print(prefix, json.dumps(found, default=str))
    for k, v in obj.items():
        if isinstance(v, dict) and k not in fields:
            concise_results(v, prefix + "/" + str(k))

if "cross_ne_pooled_results" in globals():
    concise_results(cross_ne_pooled_results)

print("\n=== SELECTED PERSISTENT OBJECT NAMES ===")
for k, v in list(globals().items()):
    if any(t in k.lower() for t in
           ["rep_results", "lag_results", "pooled_results", "lag_profile",
            "cross_ne", "insulin_rep", "joint_insulin"]):
        if isinstance(v, dict):
            print(k, "dict keys:", list(v)[:12])
        elif hasattr(v, "shape"):
            print(k, "shape:", tuple(v.shape))

print("\n=== DATA AND HISTORY SCHEMA ===")
for k in ["train", "tuning", "panel", "panel_tune", "Hrich"]:
    obj = globals().get(k)
    if obj is not None:
        print(k, "shape:", obj.shape,
              "columns:", list(obj.columns)[:35])
print("EVAL_IDS preview:", list(EVAL_IDS)[:8])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python run 26: recovered validation\n"
        "- Existing edit: pooled tuning factual NLL increase +0.00030707 "
        "over 400 stays; well within +0.005.\n"
        "- Existing dose-edit test gave exact zero for all four readouts.\n"
        "- Reviewing independent replications of nominal NE curvature and "
        "delayed insulin associations before accepting a zero target.\n"
        "- No weight changes. Forward passes: 7.\n"
    )
```
