One helper named “set encoder” actually encodes individual atoms and cannot handle an empty set. That explains the failed diagnostic, though not yet the earlier posterior mismatch. I’ll replace it with the architecture’s formula and check atom kind explicitly. Near-zero posterior error would validate the reconstruction; persistent error would refute this explanation.

```python
import itertools

def clean_single_encode(atoms, prefix):
    hidden = mlp(atoms, prefix + ".mlp1")
    size = linear(torch.full((len(atoms),1), np.log(2), dtype=torch.float32),
                  prefix + ".size")
    return layer_norm(mlp(hidden + size, prefix + ".mlp2"), prefix + ".norm")

clean_empty_obs = layer_norm(
    mlp(linear(torch.zeros((1,1)), "encode.size"), "encode.mlp2"), "encode.norm")
print("empty observation vs cached:", float((clean_empty_obs-e_empty[:1]).square().mean()))

kind_search = {}
for ev in ["insulin", "norepinephrine", "map_cuff", "glucose_finger",
           "map_arterial", "glucose_lab"]:
    ids1 = np.asarray(isolated_by_event[ev], dtype=int)[:12]
    recs1 = [isolated_record(int(i), ev) for i in ids1]
    bas1 = torch.stack([make_atom(rec) for rec in recs1])
    cx1 = state_context(ids1)
    truth1 = sx[ids1,960:]
    drug1 = ev in ("insulin","norepinephrine")
    if drug1:
        enc1 = clean_single_encode(bas1, "encode_action")
        zp1 = posterior_mean(cx1, clean_empty_obs.expand(len(ids1),-1), enc1)
        print(ev, "original helper MSE", float((zp1-truth1).square().mean()))
        continue
    oldunit = 2 if ev.startswith("map_") else 5
    cfgs1 = list(itertools.product(range(3), range(6), range(5), range(4)))
    deltas1 = torch.stack([
        W["embed.kind.weight"][ak]-W["embed.kind.weight"][0]
        +W["embed.categorical.unit.weight"][un]-W["embed.categorical.unit.weight"][oldunit]
        +W["embed.ekind.weight"][ek]-W["embed.ekind.weight"][1]
        +W["embed.kq.weight"][kq]-W["embed.kq.weight"][0]
        for ak,un,ek,kq in cfgs1])
    enc1 = clean_single_encode(
        (bas1[None]+deltas1[:,None]).reshape(-1,192), "encode")
    zp1 = posterior_mean(cx1.repeat(len(cfgs1),1), enc1, torch.zeros_like(enc1))
    errs1 = (zp1.reshape(len(cfgs1),len(ids1),32)-truth1[None]).square().mean((1,2))
    order1 = torch.argsort(errs1)[:4].tolist()
    kind_search[ev] = [(float(errs1[j]), cfgs1[j]) for j in order1]
    print(ev, kind_search[ev])
print("bin-zero norms:", {a:float(W["embed.bins."+a+".weight"][0].norm())
                          for a in ("value","dose","rate")})
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 33; forwards 3. Corrected an inspection helper that did not represent empty sets. Testing WORLD/REVEAL/END kind explicitly with clean single-atom encoding. No edits.\n")
```