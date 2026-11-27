The state tool returned complete bundle histories for 100 training stays. I’ll inspect their structure and translate the record-based impulse responses into the exact rollout readouts. That will give us numerical targets before choosing an edit. Similar targets across training and tuning stays would support using them; a large discrepancy would call for subgroup analysis.

```python
state_file = np.load("/work/fwd_3/states_original.npz")
states0 = {k: state_file[k] for k in state_file.files}
print("=== STATE ARRAYS ===")
for k, v in states0.items():
    print(k, v.shape, v.dtype)
print(pd.DataFrame({k: v[:12] for k, v in states0.items() if v.ndim == 1}).to_string(index=False))
print("State component SD ranges:")
for label, sl in [("out", slice(0,192)), ("h", slice(192,960)), ("z", slice(960,992))]:
    sd = states0["x"][:, sl].std(axis=0)
    print(label, np.quantile(sd, [0, .1, .5, .9, 1]).round(5))

def make_plain_panel(records, ids):
    panel_idx = pd.MultiIndex.from_product(
        [ids, np.arange(-1,25)], names=["stay_id", "t"]
    )
    m = records.loc[records.event.isin(measurements)].pivot_table(
        index=["stay_id","hour"], columns="event", values="value", aggfunc="first"
    )
    m.index = m.index.set_names(["stay_id","t"])
    a = records.loc[records.event.isin(drugs), ["stay_id","hour","event","rate"]].copy()
    a["t"] = np.floor(a.hour).astype(int)
    a = a.pivot_table(index=["stay_id","t"], columns="event", values="rate", aggfunc="sum")
    p = m.reindex(panel_idx).join(a)
    p[drugs] = p[drugs].fillna(0)
    return p

tune_panel = make_plain_panel(tuning, tuning_ids)

def response_arrays(panel, ids, drug, kernel):
    dose = panel[drug].to_numpy().reshape(len(ids),26)
    response = np.zeros_like(dose)
    reached = np.zeros_like(dose, dtype=bool)
    for h in range(13,25):
        for s in range(12,h):
            response[:,h+1] += kernel[h-s-1] * dose[:,s+1]
            reached[:,h+1] |= dose[:,s+1] > 0
    return response, reached

record_targets = {}
for drug, outcome, edit in [
    ("insulin","glucose_finger","insulin_x2"),
    ("norepinephrine","map_arterial","ne_x2")
]:
    kernel = np.array(impulse_results[drug]["kernel"])
    record_targets[drug] = {}
    for label, panel, ids in [
        ("train",wide,train_ids), ("tuning_all",tune_panel,tuning_ids)
    ]:
        response, reached = response_arrays(panel, ids, drug, kernel)
        observed = panel[outcome].notna().to_numpy().reshape(len(ids),26)
        use = observed & reached
        target = float(response[use].mean())
        record_targets[drug][label] = {"x2":target, "half":-0.5*target, "n":int(use.sum())}
        print("Record-based target", drug, label, record_targets[drug][label])
    response, reached = response_arrays(tune_panel,tuning_ids,drug,kernel)
    fwd = pd.read_csv("/work/fwd_2/dose_edit_original.csv")
    rows = fwd[(fwd.edit == edit) & fwd.reached]
    row_i = np.searchsorted(tuning_ids, rows.stay_id.to_numpy())
    vals = response[row_i, rows.hour.to_numpy(int)+1]
    record_targets[drug]["baseline_exact_rows"] = {
        "x2":float(vals.mean()), "half":float(-.5*vals.mean()), "n":len(vals)
    }
    print("Exact forward readout rows", drug, record_targets[drug]["baseline_exact_rows"])

print("\n=== EDIT-RELEVANT TENSORS ===")
for name in NAMES:
    if (
        name.startswith(("embed.", "encode_action.", "z_", "gap.", "query_norm.", "type_emb.",
                         "attr_heads.value", "last_numeric.", "last_status.", "embed_transition.value"))
        or any(s in name for s in ("core.prelude", "core.coda", "core.b_in", "core.out_norm",
                                  "core.delta_in", "core.a_in", "core.c_out", "core.z_in"))
    ):
        w = load_original(name)
        print(name, SHAPES[name], "RMS", round(float(np.sqrt(np.mean(w*w))),5))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward request 3 / Run 6\n"
        "Original all-bundle states for training IDs 1010000000..99: 4035 rows. "
        "Computed empirical-kernel readout targets using fixed recorded doses "
        "and the exact reached-hour convention; half-dose response is -1/2 "
        "the doubling response under this linear structural approximation.\n"
        + json.dumps(record_targets, indent=2) + "\n"
    )
```