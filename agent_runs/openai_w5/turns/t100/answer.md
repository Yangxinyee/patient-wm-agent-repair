The saved routines can support a controlled action-path test, with no weight changes. I’ll first check that their recurrent step and readout work on a small sample. Agreement with saved states and repeatable readouts would support using local sensitivities to screen candidates; large discrepancies would make that unsafe.

```python
print("Attention conventions:",
      {k: globals().get(k) for k in ["wm_memory_mode", "wm_kv_layout", "head_layout_spec"]})

drug_rows88 = []
for j in np.flatnonzero(s4["has_insulin"] | s4["has_norepinephrine"]):
    if j == 0 or s4["stay_id"][j] != s4["stay_id"][j-1]:
        continue
    rr = np.flatnonzero(np.asarray(cacheTrain["previous"]) == j)
    if len(rr):
        events = cacheTrain["df"].iloc[rr]["event"].tolist()
        if "glucose_finger" in events or "map_arterial" in events:
            drug_rows88.append(int(j))

probe_rows88 = drug_rows88[::max(1, len(drug_rows88)//6)][:6]
print("Available drug-to-reading bundles:", len(drug_rows88), "probes:", probe_rows88)

for j in probe_rows88[:2]:
    z = torch.as_tensor(s4["x"][j, 960:992]).reshape(1, 32)
    hb = local_step67(j, z, include_current=True, age_mode="input",
                      age_units=3600.0, age_scale=1.0)
    print("Bundle", j, "step return", type(hb).__name__, getattr(hb, "shape", None))
    if isinstance(hb, torch.Tensor):
        hs = torch.as_tensor(s4["x"][j, 192:960]).reshape(1,4,192)
        print("State RMS discrepancy", float(((hb-hs)**2).mean().sqrt()))
        nm = next_means_crn74(j, hb, nsamples=16, seed=913)
        print("Readout types/shapes",
              [(type(q).__name__, getattr(q, "shape", None)) for q in nm])
        print("Readout", nm)

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 84–88: action-path screening\n"
            "- Re-read architecture; no edits retained.\n"
            "- Run 83's counterfactual decoder introduced excessive factual variation and is rejected.\n"
            "- Checking the direct action/recurrent response with fixed states and common random numbers.\n")
```
