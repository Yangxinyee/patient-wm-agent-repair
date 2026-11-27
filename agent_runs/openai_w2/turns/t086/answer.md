The END convention is now identified: it enters the observation encoder, with dose, rate and units absent. That reproduces the posterior to about \(4\times10^{-5}\). I’ll reconstruct all posterior features and their responses to dose changes next. Accurate reconstruction across bundle types is the prerequisite for a residual-dose edit.

```python
def embed_end_repair(row):
    rr = dict(row)
    rr["value"], rr["dose"], rr["rate"] = np.nan, np.nan, np.nan
    a = embed_record_atom(rr, kind1=2, end_mode1="same")
    w = Wrepair["embed.categorical.unit.weight"]
    a = a-w[unit_ids_repair[rr["event"]]]+w[0]
    w = Wrepair["embed.categorical.rate_unit.weight"]
    return a-w[rate_unit_ids_repair[rr["event"]]]+w[0]

def encode_variable_sets_repair(atom_sets, prefix, zero_empty=False):
    output = torch.zeros((len(atom_sets),192),dtype=torch.float32)
    counts = np.array([len(a) for a in atom_sets])
    for n in np.unique(counts):
        ii = np.flatnonzero(counts==n)
        if n==0:
            if not zero_empty:
                output[ii] = encode_atom_set([],prefix)
        else:
            aa = torch.stack([torch.stack(atom_sets[i]) for i in ii])
            output[ii] = batched_set_repair(aa,prefix)
    return output

posterior_rows_repair = np.array([
    j for j in range(1,len(sm_repair))
    if sm_repair.iloc[j]["stay_id"]==sm_repair.iloc[j-1]["stay_id"]
])
posterior_ctx_repair, posterior_hflow_repair = previous_context_repair(posterior_rows_repair)
obs_sets_repair, action_records_repair = [], []
dose_actual_repair = np.zeros((len(posterior_rows_repair),2),dtype=np.float32)

for t,j in enumerate(posterior_rows_repair):
    key = (int(sm_repair.iloc[j]["stay_id"]),round(float(sm_repair.iloc[j]["hour"]),4))
    starts = start_lookup_repair.get(key,[])
    ends = end_lookup_repair.get(key,[])
    obs_sets_repair.append(
        [embed_record_atom(r) for r in starts
         if r["event"] not in ("insulin","norepinephrine")]
        + [embed_end_repair(r) for r in ends]
    )
    actions = [r for r in starts if r["event"] in ("insulin","norepinephrine")]
    action_records_repair.append(actions)
    for r in actions:
        d = 0 if r["event"]=="insulin" else 1
        dose_actual_repair[t,d] += float(r["rate"])

posterior_e_repair = encode_variable_sets_repair(obs_sets_repair,"encode")
posterior_u_repair = encode_variable_sets_repair(
    [[embed_record_atom(r) for r in a] for a in action_records_repair],
    "encode_action",zero_empty=True
)
posterior_phi_repair = tF.gelu(rlinear(torch.cat([
    posterior_ctx_repair,posterior_e_repair,posterior_u_repair
],1),"z_post.0"))
posterior_z_repair = rlinear(posterior_phi_repair,"z_post.2")[:,:32]
posterior_error_repair = (
    (posterior_z_repair-sx_repair[posterior_rows_repair,960:992])**2
).mean(1).sqrt().numpy()
audit = pd.DataFrame({
    "action":dose_actual_repair.sum(1)>0,
    "rmse":posterior_error_repair
})
print("Posterior reconstruction by bundle type:")
print(audit.groupby("action")["rmse"].agg(["count","mean","median","max"]))
print("Worst rows:")
worst = np.argsort(posterior_error_repair)[-6:]
print(sm_repair.iloc[posterior_rows_repair[worst]].assign(
    rmse=posterior_error_repair[worst]).to_string(index=False))

posterior_cf_phi_repair, posterior_cf_targets_repair = {}, {}
for drug,d in (("insulin",0),("norepinephrine",1)):
    for multiplier in (0.5,2.0):
        action_sets = []
        for records in action_records_repair:
            atoms = []
            for r in records:
                rr = dict(r)
                if rr["event"]==drug:
                    rr["rate"] *= multiplier
                    rr["dose"] *= multiplier
                atoms.append(embed_record_atom(rr))
            action_sets.append(atoms)
        uc = encode_variable_sets_repair(action_sets,"encode_action",zero_empty=True)
        phic = tF.gelu(rlinear(torch.cat([
            posterior_ctx_repair,posterior_e_repair,uc
        ],1),"z_post.0"))
        posterior_cf_phi_repair[(drug,multiplier)] = phic.numpy()
        target = np.zeros_like(dose_actual_repair)
        target[:,d] = dose_actual_repair[:,d]*(multiplier-1)
        posterior_cf_targets_repair[(drug,multiplier)] = target
        active = dose_actual_repair[:,d]>0
        print(drug,multiplier,"active",int(active.sum()),
              "hidden RMS change",float(
                  (phic[active]-posterior_phi_repair[active]).square().mean().sqrt()
              ))

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Runs 70–72: END routing resolved\n"
        "Single-start posterior RMSE 2.66e-5. END atoms belong to the observation "
        "encoder, with numeric attributes and units absent, not to the action "
        "encoder. This convention gives END-bundle posterior RMSE 4.1e-5. "
        "Constructing original and synthetic-dose posterior hidden features; "
        "no weights edited.\n"
    )
```
