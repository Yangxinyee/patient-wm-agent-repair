Active-drug posteriors now match to about \(5\times10^{-7}\), but reading posteriors do not. That validates the drug inputs and time context, not the observation encoding. I’ll isolate the discrepancy by comparing atom construction and errors by bundle size. Only numerical agreement on readings will justify proceeding.

```python
from collections import Counter

print("=== FRESH VERSUS RETAINED ATOM CONSTRUCTION ===")
for code in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
             "insulin", "norepinephrine"]:
    a = next(a for a in rr_desc4 if a["code"] == code and not a["end"])
    old = r95_base_atom(**a).clone()
    if code in rr_sensors:
        old_unit = 2 if code.startswith("glucose") else 3
        old += (rr_original["embed.categorical.unit.weight"][rr_units[code]]
                - rr_original["embed.categorical.unit.weight"][old_unit])
    else:
        old += rr_original["embed.categorical.unit.weight"][
            3 if code == "insulin" else 4
        ]
    fresh = rr_atom(**a)
    print(code, "max difference", float((fresh - old).abs().max()),
          "value/rate/dose", a["value"], a["rate"], a["dose"])

rr_e_mask4 = rr_routes4[rr_best4[1][0]][0]
rr_e_count4 = np.bincount(
    rr_owner4[torch.tensor(rr_e_mask4)].numpy(), minlength=rr_n4
)
print("=== POSTERIOR ERROR BY OBSERVATION COUNT ===")
for reading in [False, True]:
    for count in sorted(np.unique(rr_e_count4)):
        mask = rr_valid4 & (rr_has_reading4 == reading) & (rr_e_count4 == count)
        if mask.any():
            print("reading", reading, "count", int(count), "n", int(mask.sum()),
                  "median/max", np.quantile(rr_zerror4[mask], [.5, 1]).tolist())

print("=== LARGEST ERRORS ===")
for i in np.argsort(rr_zerror4 * rr_valid4)[-5:]:
    print(rr_keys4[i], float(rr_zerror4[i]),
          [(a["code"], a["value"], a["end"]) for a in rr_bundles4[i]])

print("=== VOCABULARY KIND DEFINITIONS ===")
print("event_kinds", rr_vocab["event_kinds"])
print("atom_kinds", rr_vocab["atom_kinds"])

print("=== RETAINED ATOM NUMERIC CONSTRUCTION ===")
import dis
for inst in dis.Bytecode(r95_base_atom):
    if inst.offset < 400:
        print(inst.offset, inst.opname, inst.argrepr)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 104\nFresh reconstruction gives numerical agreement for active-drug "
        "posteriors (max RMSE 5.5e-7), but not reading posteriors. Drug/time paths "
        "validated; observation path remains invalid. Checking atom construction "
        "and error dependence on bundle size. No edits.\n"
        "Counts: 104 Python runs, 6 forward passes.\n"
    )
```
