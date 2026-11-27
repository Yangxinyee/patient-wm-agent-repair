The posterior is small enough for a low-rank correction. Before choosing its direction, I’ll measure how its latent state responds to recorded glucose and MAP readings, conditional on preceding context, and inspect the action-fit arrays. Consistent directions across the two sensors would support a shared physiological correction; inconsistent directions would argue against it.

```python
from sklearn.linear_model import Ridge
print("=== ACTION FIT ARRAYS ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    if key.startswith("r41") or any(s in key.lower() for s in ["aug_", "expected_model", "policy_model"]):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(key, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(key, [(str(k), getattr(v, "shape", type(v).__name__)) for k,v in list(obj.items())[:6]])
        elif hasattr(obj, "coef_"):
            print(key, type(obj).__name__, np.shape(obj.coef_))
print("encode_bundles signature:", inspect.signature(encode_bundles))
print("make_atom signature:", inspect.signature(make_atom))
print("make_atom constants:", str(make_atom.__code__.co_consts)[:1000])
print("categorical tables:", [(n, SHAPES[n]) for n in NAMES if "embed.categorical." in n])
print("action-like full arrays:", [(k, tuple(v.shape)) for k,v in list(globals().items())
      if isinstance(v, (np.ndarray, torch.Tensor)) and tuple(v.shape)==(4313,192)
      and any(s in k.lower() for s in ["u", "act"])])

# Pre-bundle context, using authoritative states.
r45_valid = np.flatnonzero((np.arange(len(S["hour"])) > 0) &
    (S["stay_id"] == np.roll(S["stay_id"], 1)))
r45_prev = torch.as_tensor(S["x"][r45_valid-1], dtype=torch.float32)
r45_dt = torch.as_tensor((S["hour"][r45_valid]-S["hour"][r45_valid-1])*3600,
                        dtype=torch.float32)
r45_hf = r45_prev[:,192:960].reshape(-1,4,192) * torch.exp(
    -r45_dt[:,None,None] * torch.exp(W["core.flow.a"]))
r45_ctx = r45_prev[:,:192] + layer_norm(r45_hf.mean(1),"core.out_norm") + mlp(
    torch.log1p(r45_dt[:,None])/10, "gap")
r45_ctx_map = {int(i): j for j,i in enumerate(r45_valid)}
r45_bundle_map = {(int(s),round(float(h),4)): i
                 for i,(s,h) in enumerate(zip(S["stay_id"],S["hour"]))}
r45_sensor_directions = {}
print("\n=== CONDITIONAL READING-WRITE DIRECTIONS ===")
for event in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]:
    rows = state_records[state_records.event.eq(event)]
    pairs = [(r45_bundle_map.get((int(r.stay_id),round(float(r.hour),4))),float(r.value))
             for r in rows.itertuples()]
    pairs = [(i,v) for i,v in pairs if i in r45_ctx_map]
    ii = np.array([i for i,v in pairs],dtype=int)
    vv = np.array([v for i,v in pairs])
    cc = np.asarray(r45_ctx)[[r45_ctx_map[i] for i in ii]]
    scale = float(np.std(vv))
    xx = np.column_stack([cc, (vv-vv.mean())/scale])
    yy = np.asarray(S["x"])[ii]
    reg = Ridge(alpha=10).fit(xx, yy)
    dz = reg.coef_[960:992,-1]/scale
    dout = reg.coef_[:192,-1]/scale
    dh = reg.coef_[192:960,-1].reshape(4,192).mean(0)/scale
    r45_sensor_directions[event] = {"z":dz,"out":dout,"h":dh,"n":len(ii)}
    print(event, "n",len(ii), "||dz/unit||",round(float(np.linalg.norm(dz)),6),
          "||dout/unit||",round(float(np.linalg.norm(dout)),6),
          "||dh/unit||",round(float(np.linalg.norm(dh)),6))
for a,b in [("glucose_finger","glucose_lab"),("map_arterial","map_cuff")]:
    va,vb = r45_sensor_directions[a]["z"],r45_sensor_directions[b]["z"]
    print(a,b,"latent-direction cosine:",
          float(va@vb/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-12)))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 45; forwards 3. Posterior MLP is 576->192->64; no edits. Estimated sensor-specific latent and state writing directions conditional on authoritative preceding context, to guide a causal channel rather than use medical assumptions.\n")
```
