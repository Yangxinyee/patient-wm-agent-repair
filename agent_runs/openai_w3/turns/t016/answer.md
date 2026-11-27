The local reconstruction matches the saved posterior means to about 10⁻¹⁰ mean-squared error for single-drug starts. That identifies the activation, units, and empty observation encoding. I’ll extend the check to infusion endings and mixed bundles, then test whether frozen posterior features can decode dose deviations. Good held-out decoding would support a targeted residual-dose edit.

```python
from collections import defaultdict

start_atoms = defaultdict(list)
end_atoms = defaultdict(list)
for r in state_records.itertuples(index=False):
    start_atoms[(int(r.stay_id), round(float(r.hour), 4))].append(r)
    if np.isfinite(r.end_hour):
        end_atoms[(int(r.stay_id), round(float(r.end_hour), 4))].append(r)

def make_atom(r, is_end=False, end_mode="preserve"):
    ev = r.event
    cid = code_ids[ev]
    ek = 2 if ev in ["insulin", "norepinephrine"] else \
         1 if ev in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"] else \
         3 if ev == "hospital_admission" else 4
    numbers = np.zeros(6, dtype=np.float32)
    bins = {"value": 0, "dose": 0, "rate": 0}
    for j, attr in enumerate(["value", "dose", "rate"]):
        val = getattr(r, attr)
        if np.isfinite(val) and (not is_end or end_mode == "preserve"):
            tab = numeric_tables[(ev, attr)]
            numbers[2*j] = np.clip((float(val)-tab["median"])/tab["iqr"], -10, 10)
            numbers[2*j+1] = 1
            bins[attr] = 1 + np.sum(float(val) > np.asarray(tab["fine_bin_edges"]))
    unit = 3 if ev == "insulin" else 4 if ev == "norepinephrine" else \
           2 if ev.startswith("map_") else 5 if ev.startswith("glucose_") else 0
    rateunit = 2 if ev == "insulin" else 3 if ev == "norepinephrine" else 0
    if is_end and end_mode == "clear_all":
        unit = rateunit = 0
    t = (W["embed.code.weight"][cid] + W["embed.kind.weight"][2 if is_end else 0]
         + W["embed.ekind.weight"][ek] + W["embed.kq.weight"][0]
         + W["embed.categorical.unit.weight"][unit]
         + W["embed.categorical.rate_unit.weight"][rateunit]
         + linear(torch.tensor(numbers), "embed.numeric"))
    for attr, b in bins.items():
        t = t + W[f"embed.bins.{attr}.weight"][b]
    return t

def encode_bundles(end_mode="preserve", action_scale=1.0):
    atom_vectors, atom_bundle, action_flags = [], [], []
    bundle_atom_info = []
    for i, (sid, hour) in enumerate(zip(S["stay_id"], S["hour"])):
        key = (int(sid), round(float(hour), 4))
        atoms_here = [(r, False) for r in start_atoms[key]] + \
                     [(r, True) for r in end_atoms[key]]
        bundle_atom_info.append(atoms_here)
        for r, is_end in atoms_here:
            atom_vectors.append(make_atom(r, is_end, end_mode))
            atom_bundle.append(i)
            action_flags.append(r.event in ["insulin", "norepinephrine"])
    atom_vectors = torch.stack(atom_vectors)
    atom_bundle = torch.tensor(atom_bundle, dtype=torch.long)
    action_flags = torch.tensor(action_flags, dtype=torch.bool)
    result = []
    for is_action, prefix in [(False, "encode"), (True, "encode_action")]:
        sel = action_flags == is_action
        sums = torch.zeros(len(sx), 192)
        sums.index_add_(0, atom_bundle[sel], mlp(atom_vectors[sel], prefix+".mlp1"))
        n = torch.bincount(atom_bundle[sel], minlength=len(sx)).float()
        t = sums / torch.sqrt(n.clamp_min(1))[:,None] + \
            linear(torch.log1p(n)[:,None], prefix+".size")
        t = layer_norm(mlp(t, prefix+".mlp2"), prefix+".norm")
        if is_action:
            t[n == 0] = 0
        result.append(t)
    return result[0], result[1], bundle_atom_info

valid_idx = np.where(same_stay)[0]
dt_all = torch.tensor((S["hour"][valid_idx]-S["hour"][valid_idx-1])*3600,
                      dtype=torch.float32)
hflow_all = sx[valid_idx-1,192:960].reshape(-1,4,192) * \
            torch.exp(-dt_all[:,None,None]*torch.exp(W["core.flow.a"])[None,None,:])
ctx_all = (sx[valid_idx-1,:192] + layer_norm(hflow_all.mean(1),"core.out_norm")
           + mlp(torch.log(dt_all.clamp_min(1))[:,None]/10,"gap"))
end_checks = []
for end_mode in ["preserve", "clear_numeric", "clear_all"]:
    ee, uu, bundle_atom_info = encode_bundles(end_mode)
    pp = mlp(torch.cat([ctx_all,ee[valid_idx],uu[valid_idx]],1),"z_post")[:,:32]
    mse_rows = ((pp-sx[valid_idx,960:992])**2).mean(1).numpy()
    end_checks.append((float(mse_rows.mean()),end_mode))
    print("END MODE",end_mode,"MSE all:",mse_rows.mean(),
          "median:",np.median(mse_rows),"max:",mse_rows.max(),
          "drug MSE:",mse_rows[drugs_mask[valid_idx]].mean())
best_end_mode = min(end_checks)[1]
e_all,u_all,bundle_atom_info = encode_bundles(best_end_mode)
posterior_inputs = torch.cat([ctx_all,e_all[valid_idx],u_all[valid_idx]],1)
post_hidden = torch.nn.functional.gelu(linear(posterior_inputs,"z_post.0"))
print("BEST END MODE:",best_end_mode)
print("Posterior feature SD quantiles:",
      np.quantile(post_hidden.numpy().std(0),[0,.1,.5,.9,1]))
print("Posterior z mean and SD:",
      float(sx[:,960:992].mean()),float(sx[:,960:992].std()))

# Pre-dose policy predictions already estimated from training records.
dose_labels = np.zeros((len(sx),2))
innovation_labels = np.zeros_like(dose_labels)
for i, atoms_here in enumerate(bundle_atom_info):
    for r,is_end in atoms_here:
        if not is_end and r.event in ["insulin","norepinephrine"]:
            j = ["insulin","norepinephrine"].index(r.event)
            hour = int(np.floor(r.hour))
            dose_labels[i,j] = r.rate
            if 0 <= hour < 24:
                row = int(np.searchsorted(train_ids,r.stay_id))*24 + hour
                innovation_labels[i,j] = r.rate-dose_predictions_full[r.event][row]
print("Drug starts:",(dose_labels>0).sum(0))
print("Dose innovation mean/SD among starts:",
      [(float(innovation_labels[dose_labels[:,j]>0,j].mean()),
        float(innovation_labels[dose_labels[:,j]>0,j].std())) for j in range(2)])

with open("/work/notes.md","a") as f:
    f.write("\nPython runs: 12; forwards: 3. Matched single-start z posterior "
            "locally to ~1e-10 MSE. Action units: insulin unit=3/rate_unit=2; "
            "NE unit=4/rate_unit=3. MLP activation GELU; observation empty "
            "bundle is encoded (not zero). Full posterior END reconstruction: "
            + str(end_checks) + ". No edits yet.\n")
```
