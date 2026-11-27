The original rollouts substantially understate insulin’s estimated effect, and doubled norepinephrine moves MAP in the opposite direction to the record-based estimate. A correction is warranted, but factual fit remains the constraint. I’ll check the register inputs and state alignment next; systematic errors tied to either would explain the reconstruction mismatch.

```python
print("=== RECONSTRUCTION INPUTS ===")
print("head_meta columns:", head_meta.columns.tolist())
print(head_meta.head(10).to_string(index=False))
print("nf first rows:\n", nf[:8])
print("best config:", best_reconstruction["config"])
print("helper defaults:", {
    fn: globals()[fn].__defaults__
    for fn in ("wlin", "wnorm", "wmlp", "actfn")
})
print("numeric note:", VOCAB.get("numeric_note"))
print("value note:", VOCAB.get("value_bin_note"))
for key, spec0 in numeric_specs.items():
    print(key, spec0)

print("\n=== RELEVANT EMBEDDINGS ===")
for n in NAMES:
    if n.startswith(("embed.", "last_", "type_emb", "core.out_norm", "query_norm")):
        print(n, SHAPES[n])
print("last_numeric column norms:",
      torch.linalg.vector_norm(W0["last_numeric.weight"], dim=0).numpy())
print("last_numeric column inner products:\n",
      np.round((W0["last_numeric.weight"].T @ W0["last_numeric.weight"]).numpy(), 2))
print("status norms:", torch.linalg.vector_norm(W0["last_status.weight"], dim=1).numpy())

print("\n=== STATES FILE ===")
with np.load("/work/fwd_3/states_original.npz") as zfile:
    for key in zfile.files:
        ar = zfile[key]
        print(key, ar.shape, ar.dtype, repr(ar[:4])[:400])

print("\n=== ORIGINAL ROLLOUT SUMMARY ===")
print(Path("/work/fwd_2/summary.json").read_text())

print("\n=== RECONSTRUCTION ERROR BY REGISTER AGE ===")
probe = hm.copy()
probe["error"] = np.asarray(check["mean"]) - target_means
for c in ("event", "age", "last_age", "dt", "last_value", "last_bin"):
    if c in probe:
        if c == "event":
            print(probe.groupby(c)["error"].agg(["mean", "std", "count"]))
        else:
            print(c, probe[[c, "error"]].corr().iloc[0, 1])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 15\nOriginal rollout on the initial tuning sample: reached-hour "
            "means insulin x2 -1.85, half +0.78; norepinephrine x2 -0.50, half -0.60. "
            "These are substantially weaker than controlled record-based estimates "
            "(and NE x2 has the wrong sign). Still no edits.\n")
```