Changing units alone did not produce a close match. I’ll test whether the general atom helper is incorrectly adding embeddings for absent attributes, using the already-matched drug-start helper as a reference. If removing those terms fixes both drug starts and readings, that would explain the mismatch.

```python
try:
    print("=== RECORD ATOM SOURCE ===")
    print(inspect.getsource(recon_record_atom)[:4500])
except (OSError, TypeError):
    print("Record-atom source is unavailable; testing its output directly.")

print("\n=== DRUG START ATOM COMPARISON ===")
for drug in DRUGS:
    row0 = next(tune[tune.event == drug].itertuples(index=False))
    matched = recon_atom(drug, [row0], unit_id=RECON_UNITS[drug],
                         rate_unit_id=RECON_RATE_UNITS[drug], kq_id=0)[0]
    general = recon_record_atom(row0)
    delta0 = general - matched
    print(drug, "difference norm", float(delta0.norm()),
          "after removing absent value-bin",
          float((delta0-W0["embed.bins.value.weight"][0]).norm()))

absence_search = {}
for code in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    ii = []
    for i in recon_valid_rows:
        if len(recon_obs_lists[i]) == 1 and not recon_action_lists[i]:
            row = state_df.iloc[i]
            sub = tune[(tune.stay_id == row.stay_id) & (np.abs(tune.hour-row.hour) < 1e-5)]
            if len(sub) == 1 and sub.iloc[0].event == code:
                ii.append(int(i))
        if len(ii) >= 60:
            break
    atoms = torch.stack([recon_obs_lists[i][0] for i in ii])
    choices1 = list(itertools.product(range(6), [-1, 0, 1], [-1, 0, 1], [-1, 0]))
    delta1 = torch.stack([
        W0["embed.categorical.unit.weight"][u] -
        W0["embed.categorical.unit.weight"][RECON_UNITS[code]] +
        bd * W0["embed.bins.dose.weight"][0] +
        br * W0["embed.bins.rate.weight"][0] +
        ru * W0["embed.categorical.rate_unit.weight"][0]
        for u, bd, br, ru in choices1
    ])
    ee1 = single_encode_batch((atoms[None]+delta1[:, None]).reshape(-1, 192))
    cc1 = recon_context[ii].repeat(len(choices1), 1)
    pp1 = recon_mlp(torch.cat([cc1, ee1, torch.zeros_like(ee1)], dim=1),
                    POST_PREFIX)[:, :32].reshape(len(choices1), len(ii), 32)
    target1 = torch.as_tensor(S0["x"][ii, 960:992])
    loss1 = ((pp1-target1[None])**2).mean(dim=(1, 2)).sqrt().numpy()
    jj = np.argsort(loss1)[:3]
    absence_search[code] = [(choices1[k], float(loss1[k])) for k in jj]
    print(code, "(unit, dose-bin offset, rate-bin offset, rate-unit offset):",
          absence_search[code])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 24: absent-attribute audit\n"
            + str(absence_search) +
            "\nNo edits. Python runs: 24; forward passes: 3.\n")
```
