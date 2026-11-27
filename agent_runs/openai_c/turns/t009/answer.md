The horizon estimates suggest substantial accumulation, especially for insulin. The norepinephrine persistence fit is less exact, so its target remains more uncertain. My first target calculation also included hours without the scored sensor; I’ll correct that. Then I’ll test whether dose-bin embeddings cause the model’s irregular responses, preserving numerical doses and the absent-bin embedding.

```python
PYTHON_RUNS += 1
from scipy.linalg import helmert

# Match the tool: only recorded glucose_finger / map_arterial stay-hours.
def record_based_rollouts(p, ids=None, parameters=None):
    parameters = persistence if parameters is None else parameters
    if ids is not None:
        p = p.loc[p.index.get_level_values("stay_id").isin(ids)]
    result = {}
    for drug, target, edits in [
        ("insulin", "glucose_finger", [("insulin_x2", 2.), ("insulin_half", .5)]),
        ("norepinephrine", "map_arterial", [("ne_x2", 2.), ("ne_half", .5)])
    ]:
        rates = p[drug].unstack("t").reindex(columns=np.arange(-1,25)).to_numpy()
        observed = p[target].unstack("t").reindex(
            columns=np.arange(-1,25)).notna().to_numpy()
        beta, rho = parameters[drug]["beta"], parameters[drug]["rho"]
        for edit, multiplier in edits:
            shift = np.zeros_like(rates)
            reached = np.zeros_like(rates, dtype=bool)
            for hour in range(24):
                cur, nxt = hour+1, hour+2
                dc = (multiplier-1)*rates[:,cur] if hour >= 12 else 0
                shift[:,nxt] = rho*shift[:,cur] + beta*dc
                reached[:,nxt] = reached[:,cur] | (
                    (rates[:,cur] > 0) if hour >= 12 else False)
            use = reached[:,14:26] & observed[:,14:26]
            result[edit] = dict(mean=float(shift[:,14:26][use].mean()),
                                n_stay_hours=int(use.sum()),
                                n_stays=int(use.any(1).sum()))
    return result

target_probe = record_based_rollouts(panel_tuning, probe_ids)
target_train = record_based_rollouts(panel)
print("=== CORRECTED RECORD-BASED TARGETS ===")
print("Probe:", target_probe)
print("Training population:", target_train)
print("Tool reached counts:",
      original_rollout[original_rollout.reached].groupby("edit").agg(
          n_stay_hours=("shift","size"), n_stays=("stay_id","nunique")).to_dict())

# Estimate dose-innovation state directions, without using model weights.
sx = states_original["x"].astype(float)
ss = states_original["stay_id"]
sh = states_original["hour"]
st = np.floor(sh+1e-5).astype(int)
keys = pd.MultiIndex.from_arrays([ss, st], names=["stay_id","t"])
sc = controls.reindex(keys).fillna(0)
sd = panel[DRUGS].reindex(keys).fillna(0).to_numpy()
previous = np.arange(len(sx))-1
same_stay = np.r_[False, ss[1:] == ss[:-1]]
previous[~same_stay] = np.arange(len(sx))[~same_stay]
before = sx[previous]
delta_state = sx-before
valid_actions = (same_stay & (st >= 0) & (st <= 23) &
                 (sd.sum(1) > 0) &
                 (states_original["has_insulin"] |
                  states_original["has_norepinephrine"]))

dose_models = {}
predicted_dose_states = np.zeros_like(sd)
eligible = (t >= 0) & (t < 24) & (sid >= 1010000100)
fit_rows = np.flatnonzero(eligible)
fit_rows = np.random.default_rng(810).choice(
    fit_rows, min(200000,len(fit_rows)), replace=False)
for j, drug in enumerate(DRUGS):
    reg = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=.08, max_leaf_nodes=31,
        min_samples_leaf=75, l2_regularization=10,
        max_bins=128, random_state=81+j)
    reg.fit(controls.iloc[fit_rows].to_numpy(dtype=np.float32),
            panel[drug].iloc[fit_rows].to_numpy())
    predicted_dose_states[:,j] = reg.predict(sc.to_numpy(dtype=np.float32))
    dose_models[drug] = reg

innovation = sd-predicted_dose_states
state_C = np.column_stack([
    sc.to_numpy(), before[:,:192], (sd > 0).astype(float),
    states_original["n_atoms"], sh-st
])

def state_dose_coefficients(mask, ridge=20.):
    C = state_C[mask]
    C = (C-C.mean(0)) / np.maximum(C.std(0),1e-8)
    D = innovation[mask]
    ds = np.maximum(D.std(0),1e-8)
    A = np.column_stack([np.ones(len(C)), C, D/ds])
    penalty = np.eye(A.shape[1])*ridge
    penalty[0,0] = 0
    penalty[-2:,-2:] = np.eye(2)*.01
    coef = np.linalg.solve(A.T@A+penalty, A.T@delta_state[mask])
    return coef[-2:] / ds[:,None]

dose_state_coeff = state_dose_coefficients(valid_actions)
dose_directions = {}
print("\n=== CONDITIONAL DOSE-STATE DIRECTIONS ===")
for j, drug in enumerate(DRUGS):
    parts = {
        "out": dose_state_coeff[j,:192],
        "hmean": dose_state_coeff[j,192:960].reshape(4,192).mean(0),
        "z": dose_state_coeff[j,960:992]
    }
    dose_directions[drug] = {
        name: v/np.maximum(np.linalg.norm(v),1e-12) for name,v in parts.items()
    }
    print(drug, "coefficient norms:",
          {name:round(float(np.linalg.norm(v)),5) for name,v in parts.items()})
split_coeffs = [
    state_dose_coefficients(valid_actions & ((ss-1010000000)%2 == f))
    for f in [0,1]
]
for j,drug in enumerate(DRUGS):
    a,b = split_coeffs[0][j,:192],split_coeffs[1][j,:192]
    print(drug,"split-half out-direction cosine:",
          round(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),4))
print("Between-drug out-direction cosine:",
      round(float(dose_directions["insulin"]["out"] @
                  dose_directions["norepinephrine"]["out"]),4))

# Candidate A: collapse contrasts among nonzero bins, preserving bin 0
# and the common nonzero-bin component. Numeric rate/dose inputs unchanged.
bin_contrasts = np.column_stack([np.zeros(15), helmert(16)])
clear_ops()
for name in ["embed.bins.dose.weight","embed.bins.rate.weight"]:
    for v in bin_contrasts:
        rank1(name, v.tolist(), -1., side="out")
CURRENT_NAME = "A_bin_contrasts_zero"
print("\n=== CANDIDATE A ===")
print("Operation count:",len(ops()))
print("First operation:",ops()[0])
print("Numerical embeddings and all measurement heads unchanged.")

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Run 6: corrected targets, state directions, candidate A\n"
        "- Earlier rollout targets included unrecorded sensor-hours; superseded "
        "by the sensor-observed masks here.\n"
        f"- Corrected probe targets: {target_probe!r}\n"
        f"- Corrected training-population targets: {target_train!r}\n"
        "- Estimated conditional dose-innovation directions from action-bundle "
        "state changes, controlling pre-action states and record histories. "
        "Directions saved in dose_directions; not applied yet.\n"
        "- Candidate A: 30 rank-one output projections erase contrasts between "
        "nonzero dose/rate bin rows while preserving bin 0 and the common "
        "nonzero-bin component. Numeric inputs unchanged.\n"
        "- Hypothesis: bin lookup artifacts cause irregular dose response. "
        "Require factual NLL increase <=0.005, then inspect dose rollouts.\n"
        "- Python runs: 6; forward passes: 3.\n"
    )
```
