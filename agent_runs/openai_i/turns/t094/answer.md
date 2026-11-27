The reconstruction error is smaller on action bundles but still too large on reading bundles to trust for an edit. I’ll test whether the sensor unit choices or set-pooling convention explain that pattern. A consistent convention that sharply reduces error across bundles would support the reconstruction; isolated improvements would not.

```python
print("=== EMPTY-VECTOR CHECK ===")
for label86, current86, old86 in [
    ("observation", Eworld84[obscount84-endcount84 == 0][0], latent_empty_e),
    ("action", Ubase84[actioncount84 == 0][0], empty_action58)]:
    print(label86, "RMS", float(((current86-old86)**2).mean().sqrt()))
print("Earlier helper names:")
for n86, x86 in sorted(list(globals().items())):
    if inspect.isfunction(x86) and x86.__module__ == "__main__" and (
            "encode" in n86 or n86 in ["latent_start_atom", "repair_norm"]):
        print(n86, inspect.signature(x86), str(x86.__code__.co_names)[:350])

world_tensor86 = torch.stack(world_atoms84)
world_events86 = np.array([r["event"] for b in B84 for r, end in b
                           if not end and r["event"] not in rateunit84])
world_idx86 = torch.tensor(world_indices84)
sensor_names86 = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
world_sums86 = {}
for ev86 in np.unique(world_events86):
    m86 = np.flatnonzero(world_events86 == ev86)
    a86 = world_tensor86[m86]
    world_sums86[ev86] = {}
    candidates86 = range(6) if ev86 in sensor_names86 else [0]
    for u86 in candidates86:
        av86 = (a86 + repair_weights["embed.categorical.unit.weight"][u86]
                - repair_weights["embed.categorical.unit.weight"][unit84.get(ev86, 0)])
        ss86 = torch.zeros((nb84, 192))
        ss86.index_add_(0, world_idx86[m86],
                       repair_mlp_act(av86, "encode.mlp1", "gelu"))
        world_sums86[ev86][u86] = ss86

n86 = torch.tensor(obscount84-endcount84, dtype=torch.float32)[:, None]
obs_mask86 = torch.from_numpy(valid84 & (endcount84 == 0) & (actioncount84 == 0))
units86 = {ev: unit84.get(ev, 0) for ev in world_sums86}

def encode_sum86(sums, power=.5, size_mode="log1p", counts=n86):
    if size_mode == "log1p": sizes = torch.log1p(counts)
    elif size_mode == "n": sizes = counts
    elif size_mode == "log": sizes = torch.log(counts.clamp_min(1))
    elif size_mode == "sqrt": sizes = counts.sqrt()
    elif size_mode == "log1p_div10": sizes = torch.log1p(counts)/10
    return repair_norm(repair_mlp_act(
        sums/counts.clamp_min(1).pow(power)+repair_linear(sizes, "encode.size"),
        "encode.mlp2", "gelu"), "encode.norm")

def sum_units86(units):
    return sum(world_sums86[ev][u] for ev, u in units.items())

def unit_score86(units, power=.5, size_mode="log1p"):
    e = encode_sum86(sum_units86(units), power, size_mode)
    _, z = posterior84(C84, e, U84)
    return float(((z[obs_mask86]-Ztarget84[obs_mask86])**2).mean().sqrt())

print("\n=== SENSOR-UNIT SEARCH ===")
for sweep86 in range(3):
    for ev86 in sensor_names86:
        trials86 = []
        for u86 in range(6):
            test86 = dict(units86)
            test86[ev86] = u86
            trials86.append((unit_score86(test86), u86))
        score86, units86[ev86] = min(trials86)
    print("sweep", sweep86, "units", units86, "RMS", unit_score86(units86))

pool_trials86 = []
for power86, mode86 in itertools.product(
        [0., .5, 1.], ["log1p", "n", "log", "sqrt", "log1p_div10"]):
    pool_trials86.append((unit_score86(units86, power86, mode86), power86, mode86))
pool_trials86.sort()
print("Best pooling conventions:", pool_trials86[:5])
score86, power86, mode86 = pool_trials86[0]
Eworld86 = encode_sum86(sum_units86(units86), power86, mode86)
_, z86 = posterior84(C84, Eworld86, U84)
err86 = ((z86-Ztarget84)**2).mean(1).sqrt().numpy()
for count86 in sorted(set((obscount84-endcount84)[obs_mask86.numpy()])):
    m86 = obs_mask86.numpy() & ((obscount84-endcount84) == count86)
    print("observation count", count86, "n", m86.sum(),
          "median/p95 RMS", np.quantile(err86[m86], [.5, .95]).tolist())
worst86 = np.flatnonzero(obs_mask86.numpy())
worst86 = worst86[np.argsort(err86[worst86])[-4:]]
print("Remaining worst bundles:",
      [(int(i), float(err86[i]), [r["event"] for r, end in B84[i]])
       for i in worst86])

with open("/work/notes.md", "a") as f86:
    f86.write("\n## Python 86; forward 6 — reading-feature audit\n"
              f"- Empirically selected sensor units: {units86}.\n"
              f"- Best reading-pool test: {pool_trials86[:1]}.\n"
              "- No weights changed; testing consistency across full bundles rather than relying on isolated cases.\n")
```
