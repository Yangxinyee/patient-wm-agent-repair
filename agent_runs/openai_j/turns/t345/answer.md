This is still the earlier audit, not the requested rollout, so it supplies no new evidence. I’ll check whether a new forward result exists on disk and verify the saved candidate against the installed weights. A new matching-hash result can be incorporated; otherwise, the two existing rollout groups remain our only validation of these readouts.

```python
print("=== COMPLETED FORWARD RUNS ===")
final_dirs292 = sorted(
    Path("/work").glob("fwd_*"),
    key=lambda p: int(p.name.split("_")[-1])
)
final_completed292 = []
for final_p292 in final_dirs292:
    final_sp292 = final_p292 / "summary.json"
    if not final_sp292.exists():
        continue
    final_s292 = json.loads(final_sp292.read_text())
    if final_s292.get("ok") is True:
        final_completed292.append((final_p292, final_s292))
print("Successful summaries:", len(final_completed292))
for final_p292, final_s292 in final_completed292[-8:]:
    print(final_p292.name, final_s292.get("type"),
          final_s292.get("edit_hash"), final_s292.get("n_stays"))
    final_file292 = final_p292 / (
        "dose_edit_edited.csv"
        if final_s292.get("type") == "dose_edit"
        else "factual_edited.csv"
    )
    if final_file292.exists():
        final_d292 = pd.read_csv(final_file292)
        final_ids292 = sorted(final_d292["stay_id"].unique())
        print("  stay range:", final_ids292[0], final_ids292[-1])
    final_effects292 = (
        final_s292.get("results", {}).get("edited", {}).get("edits", {})
    )
    if final_effects292:
        print("  effects:", {
            k: (v.get("mean_shift"), v.get("n_stay_hours"))
            for k, v in final_effects292.items()
        })

print("\n=== SAVED SAFETY CANDIDATE ===")
with np.load("/work/safety_candidate285.npz") as final_archive292:
    print("Archive keys:", final_archive292.files)
    for final_name292 in edited():
        if final_name292 in final_archive292.files:
            final_current292 = load(final_name292)
            if hasattr(final_current292, "detach"):
                final_current292 = final_current292.detach().cpu().numpy()
            print(final_name292, "max difference:",
                  float(np.max(np.abs(
                      np.asarray(final_current292)
                      - final_archive292[final_name292]
                  ))))

print("\n=== NON-RECOVERY NOTE HEADINGS ===")
final_sections292 = re.split(
    r"(?m)(?=^## )", Path("/work/notes.md").read_text()
)
final_headers292 = [
    s.split("\n", 1)[0] for s in final_sections292
    if s.startswith("## ")
    and not any(t in s.split("\n", 1)[0].lower()
                for t in ("recovery", "restart", "bounded"))
]
print("\n".join(final_headers292[:24]))

print("\n=== CANDIDATE-285 KERNEL METADATA ===")
final_globals292 = dict(globals())
for final_k292, final_v292 in sorted(final_globals292.items()):
    if "285" not in final_k292 or final_k292.startswith("final_"):
        continue
    if isinstance(final_v292, dict):
        print(final_k292, "dict keys:", list(final_v292)[:20])
        print("  scalars:", {
            k: v for k, v in final_v292.items()
            if isinstance(v, (str, int, float, bool))
        })
    elif isinstance(final_v292, (str, int, float, bool)):
        print(final_k292, str(final_v292)[:250])
    else:
        print(final_k292, type(final_v292).__name__,
              getattr(final_v292, "shape", ""))

with open("/work/notes.md", "a") as final_notes292:
    final_notes292.write(
        "\n## Run 292: forward-completion and candidate audit\n"
        "- Received repeated run-291 output rather than the requested "
        "third-group rollout; inspecting completed forward files.\n"
        "- Checking installed tensors against safety_candidate285.npz. "
        "No weights changed.\n"
        "- Counts: 292 Python runs; latest externally confirmed "
        "forward count is 44, pending filesystem reconciliation.\n"
    )
```