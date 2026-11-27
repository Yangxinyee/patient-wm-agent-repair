Isolated starts reconstruct to about \(3\times10^{-4}\) RMS, much smaller than their dose-induced posterior changes. The full-bundle reconstruction is still less accurate. I’ll resolve END-atom handling on drug-only bundles, where observations cannot interfere. A consistently small error there would let us fit the dose correction without rebuilding the whole model.

```python
import itertools

_kind_key99 = next(n for n in NAMES if n.startswith("embed.kind."))
_ekind_key99 = next(n for n in NAMES if n.startswith("embed.ekind."))
_kind99 = repair_weights[_kind_key99]
_ekind99 = repair_weights[_ekind_key99]
_unit99 = repair_weights["embed.categorical.unit.weight"]
_rateunit99 = repair_weights["embed.categorical.rate_unit.weight"]
_actioncat99 = repair_weights["embed.categorical.action.weight"]

_drug_cases99 = [c for c in latent_bundles if len(c) >= 3]
_end_cases99 = [c for c in _drug_cases99 if len(c[2]) > 0]
_pick99 = np.linspace(0, len(_end_cases99)-1,
                      min(64, len(_end_cases99))).round().astype(int)
_probe_cases99 = [_end_cases99[i] for i in _pick99]
_probe_indices99 = np.array([int(c[0]) for c in _probe_cases99])
_probe_ctx99 = latent_context(_probe_indices99)
_probe_target99 = torch.from_numpy(latent_state[_probe_indices99, -32:]).float()

def end_atom99(row, config):
    numbers, unit_absent, rateunit_absent, action_delta, ekind_absent = config
    r = dict(row)
    if numbers == "absent":
        for key in ("value", "rate", "dose"):
            r[key] = np.nan
    a = latent_start_atom(r) + _kind99[2] - _kind99[0]
    if unit_absent:
        uid = 3 if row["event"] == "insulin" else 4
        a = a + _unit99[0] - _unit99[uid]
    if rateunit_absent:
        a = a + _rateunit99[0] - _rateunit99[2]
    a = a + action_delta * (_actioncat99[1] - _actioncat99[0])
    if ekind_absent:
        a = a + _ekind99[0] - _ekind99[2]
    return a

def drug_u99(case, config, drug=None, multiplier=1.0):
    atoms = []
    for row in case[1]:
        r = dict(row)
        if drug is not None and r["event"] == drug:
            r["rate"] = float(r["rate"]) * multiplier
            r["dose"] = float(r["dose"]) * multiplier
        atoms.append(latent_start_atom(r))
    for row in case[2]:
        r = dict(row)
        if drug is not None and r["event"] == drug:
            r["rate"] = float(r["rate"]) * multiplier
            r["dose"] = float(r["dose"]) * multiplier
        atoms.append(end_atom99(r, config))
    return latent_action_encode(atoms)

_grid99 = []
for _cfg99 in itertools.product(("absent", "full"), (False, True),
                                (False, True), (-1, 0, 1), (False, True)):
    _u99 = torch.stack([drug_u99(c, _cfg99) for c in _probe_cases99])
    _h99 = latent_hidden(_probe_ctx99, _u99)
    _z99 = repair_linear(_h99, "z_post.2")[:, :32]
    _err99 = torch.sqrt(torch.mean((_z99 - _probe_target99)**2, dim=1))
    _grid99.append((float(_err99.mean()), float(_err99.max()), _cfg99))
_grid99.sort(key=lambda x: x[0])
print("BEST END CONVENTIONS: mean RMS, max RMS, configuration")
for _entry99 in _grid99[:6]:
    print(_entry99)
_end_config99 = _grid99[0][2]

_drug_indices99 = np.array([int(c[0]) for c in _drug_cases99])
_drug_ctx99 = latent_context(_drug_indices99)
_drug_u99 = torch.stack([drug_u99(c, _end_config99) for c in _drug_cases99])
_drug_hidden99 = latent_hidden(_drug_ctx99, _drug_u99)
_drug_z99 = repair_linear(_drug_hidden99, "z_post.2")[:, :32]
_drug_err99 = torch.sqrt(torch.mean(
    (_drug_z99 - torch.from_numpy(latent_state[_drug_indices99, -32:]))**2, dim=1))
print("\nALL DRUG BUNDLES", len(_drug_cases99),
      "mean RMS", float(_drug_err99.mean()),
      "median", float(_drug_err99.median()),
      "95th percentile", float(torch.quantile(_drug_err99, .95)))

print("\nDEVELOPMENT FACTUAL BASELINE", float(factual_base.nll.mean()))
print("Development stays:", sorted(map(int, factual_base.stay_id.unique()))[:100])

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 99 — drug-bundle reconstruction\n"
             "- Isolated starts validate to approximately 3e-4 posterior RMS.\n"
             "- Tested END conventions on drug-only bundles, selected by exposed-state "
             "agreement, and reconstructed all drug-bundle posterior hidden features.\n"
             f"- Best END configuration: {_end_config99}; drug RMS "
             f"{float(_drug_err99.mean()):.7f}. No weights changed.\n")
```