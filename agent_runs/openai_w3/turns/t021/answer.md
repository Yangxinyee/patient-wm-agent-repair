Isolated insulin and norepinephrine starts reconstruct essentially exactly; measurements and infusion endings do not. That points to input encoding rather than the context calculation. I’ll test categorical defaults and whether END atoms enter the observation or action encoder; near-zero errors across events would confirm the diagnosis.

```python
import itertools
torch.set_num_threads(8)

def encode_one_set(atom_tensor, prefix):
    x0 = mlp(atom_tensor, prefix + ".mlp1", "gelu")
    size0 = torch.full((len(atom_tensor), 1), np.log(2), dtype=torch.float32)
    return layer_norm(mlp(x0 + linear(size0, prefix + ".size"),
                          prefix + ".mlp2", "gelu"), prefix + ".norm")

empty_obs = e_empty.reshape(1, 192)
encoding_search = {}
with torch.no_grad():
    for event0 in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
        idx0 = np.array(isolated_by_event[event0][:32])
        p0 = np.array([pos_by_idx[int(i0)] for i0 in idx0])
        recs0 = [bundle_atom_info[i0][0][0] for i0 in idx0]
        atoms0 = torch.stack([make_atom(r0, False, "preserve") for r0 in recs0])
        old_unit = 2 if event0.startswith("map_") else 5
        configs0 = list(itertools.product(range(6), range(5), range(3)))
        atoms1 = torch.stack([
            atoms0 + W["embed.categorical.unit.weight"][unit0]
                   - W["embed.categorical.unit.weight"][old_unit]
                   + W["embed.ekind.weight"][ek0] - W["embed.ekind.weight"][1]
                   + W["embed.kind.weight"][kind0] - W["embed.kind.weight"][0]
            for unit0, ek0, kind0 in configs0
        ])
        n0 = len(idx0)
        enc0 = encode_one_set(atoms1.reshape(-1, 192), "encode")
        contexts0 = ctx_all[p0].repeat(len(configs0), 1)
        z0 = mlp(torch.cat([contexts0, enc0, torch.zeros_like(enc0)], 1),
                 "z_post", "gelu")[:, :32].reshape(len(configs0), n0, 32)
        err0 = ((z0 - sx[idx0, -32:])**2).mean((1, 2))
        top0 = torch.argsort(err0)[:3].tolist()
        encoding_search[event0] = [(float(err0[j0]), configs0[j0]) for j0 in top0]
        print(event0, encoding_search[event0])

    for event0 in ("insulin", "norepinephrine"):
        idx0 = np.array(isolated_by_event[event0 + " END"][:24])
        p0 = np.array([pos_by_idx[int(i0)] for i0 in idx0])
        recs0 = [bundle_atom_info[i0][0][0] for i0 in idx0]
        configs0 = list(itertools.product(
            ("preserve", "clear_numeric", "clear_all"), range(5),
            range(6), range(4), ("action", "observation")))
        all_atoms = []
        for mode0, ek0, unit0, ru0, where0 in configs0:
            old_u = 0 if mode0 == "clear_all" else (3 if event0 == "insulin" else 4)
            old_ru = 0 if mode0 == "clear_all" else (2 if event0 == "insulin" else 3)
            base0 = torch.stack([make_atom(r0, True, mode0) for r0 in recs0])
            all_atoms.append(base0 + W["embed.ekind.weight"][ek0] - W["embed.ekind.weight"][2]
                                   + W["embed.categorical.unit.weight"][unit0]
                                   - W["embed.categorical.unit.weight"][old_u]
                                   + W["embed.categorical.rate_unit.weight"][ru0]
                                   - W["embed.categorical.rate_unit.weight"][old_ru])
        atom_flat = torch.stack(all_atoms).reshape(-1, 192)
        enc_obs = encode_one_set(atom_flat, "encode").reshape(len(configs0), len(idx0), 192)
        enc_act = encode_one_set(atom_flat, "encode_action").reshape(len(configs0), len(idx0), 192)
        es, us = [], []
        for j0, cfg0 in enumerate(configs0):
            if cfg0[-1] == "action":
                es.append(empty_obs.expand(len(idx0), -1))
                us.append(enc_act[j0])
            else:
                es.append(enc_obs[j0])
                us.append(torch.zeros_like(enc_obs[j0]))
        z0 = mlp(torch.cat([ctx_all[p0].repeat(len(configs0), 1),
                           torch.stack(es).reshape(-1, 192),
                           torch.stack(us).reshape(-1, 192)], 1),
                 "z_post", "gelu")[:, :32].reshape(len(configs0), len(idx0), 32)
        err0 = ((z0 - sx[idx0, -32:])**2).mean((1, 2))
        top0 = torch.argsort(err0)[:4].tolist()
        encoding_search[event0 + " END"] = [
            (float(err0[j0]), configs0[j0]) for j0 in top0]
        print(event0 + " END", encoding_search[event0 + " END"])
with open("/work/notes.md", "a") as f:
    f.write("\nPython runs: 17; forwards: 3. Isolated drug starts match posterior to ~1e-10 MSE. Testing measurement units/kinds and END observation-vs-action routing. Results: " + repr(encoding_search) + "\n")
```