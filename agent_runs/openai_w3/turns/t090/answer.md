The probe confirms that observation bundles use the learned empty-action baseline. Measurement-unit IDs and END-atom conventions are now the remaining suspects. I’ll test them directly against the exposed inputs; near-zero error would settle the encoding without relying on assumptions about the simulator.

```python
torch.set_num_threads(8)

def encode_groups80(groups, prefix):
    n80 = len(groups)
    counts80 = torch.tensor([len(g) for g in groups], dtype=torch.float32)
    summed80 = torch.zeros(n80, 192)
    if int(counts80.sum()) > 0:
        atoms80 = torch.stack([a for g in groups for a in g])
        index80 = torch.tensor(
            [i for i, g in enumerate(groups) for _ in g], dtype=torch.long
        )
        summed80.index_add_(0, index80, mlp(atoms80, prefix + ".mlp1"))
    pooled80 = summed80 / counts80.sqrt().clamp_min(1)[:, None]
    pooled80 += linear(torch.log1p(counts80)[:, None], prefix + ".size")
    return layer_norm(mlp(pooled80, prefix + ".mlp2"),
                      prefix + ".norm").numpy()

units80 = {}
print("=== SINGLE-MEASUREMENT UNIT SEARCH ===")
for ev80 in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    idx80 = np.array([
        i for i, info in enumerate(r40_info)
        if len(info) == 1 and not info[0][1] and info[0][0].event == ev80
    ], dtype=int)
    oldunit80 = 5 if ev80.startswith("glucose") else 2
    atoms80 = torch.stack([make_atom(r40_info[i][0][0]) for i in idx80])
    errors80 = []
    for unit80 in range(6):
        trial80 = atoms80 + (
            W["embed.categorical.unit.weight"][unit80]
            - W["embed.categorical.unit.weight"][oldunit80]
        )
        pred80 = encode_one_set(trial80, "encode").numpy()
        errors80.append(float(np.mean(
            (pred80[:, e_cols79] - e_true79[idx80]) ** 2
        )))
    units80[ev80] = int(np.argmin(errors80))
    print(ev80, "n", len(idx80), "MSE by unit",
          [round(x, 9) for x in errors80], "best", units80[ev80])

def atom80(row, is_end=False, end_mode="preserve"):
    a80 = make_atom(row, is_end, end_mode)
    if row.event in units80 and not is_end:
        old80 = 5 if row.event.startswith("glucose") else 2
        a80 = a80 + W["embed.categorical.unit.weight"][units80[row.event]]
        a80 = a80 - W["embed.categorical.unit.weight"][old80]
    return a80

print("=== FULL-BUNDLE END CONVENTION SEARCH ===")
results80 = []
for mode80 in ("preserve", "clear_numeric", "clear_all"):
    for route80 in ("observation", "action"):
        eg80, ug80 = [], []
        for info80 in r40_info:
            ge80, gu80 = [], []
            for row80, end80 in info80:
                action80 = row80.event in ("insulin", "norepinephrine") and (
                    not end80 or route80 == "action"
                )
                (gu80 if action80 else ge80).append(
                    atom80(row80, end80, mode80)
                )
            eg80.append(ge80)
            ug80.append(gu80)
        ep80 = encode_groups80(eg80, "encode")
        up80 = encode_groups80(ug80, "encode_action")
        emse80 = float(np.mean((ep80[:, e_cols79] - e_true79) ** 2))
        umse80 = float(np.mean((up80[:, u_cols79] - u_true79) ** 2))
        results80.append((emse80 + umse80, mode80, route80,
                          emse80, umse80))
        print(mode80, route80, "e/u MSE", emse80, umse80)
        if len(results80) == 1 or results80[-1][0] < best80[0]:
            best80 = results80[-1]
            e80, u80 = ep80.copy(), up80.copy()

print("BEST:", best80)
with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 80; forwards 7. Tested measurement units and six END "
        f"routing/attribute conventions against exact probe coordinates. "
        f"Best units {units80}; best convention {best80}. "
        "No weight edits.\n"
    )
```