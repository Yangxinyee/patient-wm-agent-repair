The original classifier recognizes single-drug norepinephrine inputs across the tested rates, so simple dose-dependent gate failure is not supported. It can reject mixed-drug bundles, while the replacement classifier misclassifies some insulin inputs and higher gains exceed the factual budget. I’ll restrict the gate fit to recorded dose ranges and their evaluated edits, then test modest gains. A clean separation and lower factual cost would support proceeding.

```python
bounds286 = pd.read_csv("/data/train.csv", usecols=["event", "rate"])
bounds286 = bounds286[bounds286["event"].isin(["insulin","norepinephrine"])]
print("RECORD DOSE RANGES", bounds286.groupby("event")["rate"].agg(["min","max"]).to_dict())
rng286 = np.random.default_rng(286)
atoms286, actions286 = {}, {}
for drug in ("insulin","norepinephrine"):
    recorded = bounds286.loc[(bounds286["event"] == drug) & (bounds286["rate"] > 0), "rate"]
    rr = np.unique(np.r_[recorded.min()/2, recorded.max()*2,
                        np.geomspace(recorded.min()/2, recorded.max()*2, 120),
                        recorded.quantile(np.linspace(0,1,31)).to_numpy()])
    atoms286[drug] = atom_exact65(drug, np.full(len(rr), np.nan), rr, rr)
    actions286[drug] = singleton_action_repair(atoms286[drug])
ii = rng286.integers(0, len(atoms286["insulin"]), 400)
jj = rng286.integers(0, len(atoms286["norepinephrine"]), 400)
mh = (rp_mlp(atoms286["insulin"][ii], "encode_action.mlp1")
      + rp_mlp(atoms286["norepinephrine"][jj], "encode_action.mlp1")) / np.sqrt(2.0)
mh += rp_linear(torch.full((len(ii),1), float(np.log(3.0))), "encode_action.size")
mix286 = rp_norm(rp_mlp(mh, "encode_action.mlp2"), "encode_action.norm")
x286 = np.vstack([a262(actions286["norepinephrine"]), a262(mix286),
                  a262(actions286["insulin"]), np.zeros((1,192))])
y286 = np.r_[np.ones(len(actions286["norepinephrine"])+len(mix286)),
             np.zeros(len(actions286["insulin"])+1)]
gate286 = LinearSVC(C=100.0, dual=False, max_iter=30000).fit(x286, y286)
margin286 = gate286.decision_function(x286)
print("CLASSIFIER errors/positive-min/negative-max",
      int(np.sum(gate286.predict(x286) != y286)),
      float(margin286[y286 == 1].min()), float(margin286[y286 == 0].max()))

def candidate286(ins_up, ins_down, ins_gate, ne_gain, gate_scale=40.0):
    c = {n:v.clone() for n,v in base285.items()}
    c.update(asymmetric_ins279(ins_up, ins_down, ins_gate))
    w, b = a262(c["z_post.0.weight"]).copy(), a262(c["z_post.0.bias"]).copy()
    o = a262(c["z_post.2.weight"]).copy()
    gw = np.zeros(576, dtype=np.float32)
    gw[384:] = gate_scale * gate286.coef_[0]
    gb = float(gate_scale * gate286.intercept_[0])
    w[156], w[89] = gw + ne_gain*qw285/2, gw - ne_gain*qw285/2
    b[156], b[89] = gb + ne_gain*qb285/2, gb - ne_gain*qb285/2
    direction = (o[:,156]-o[:,89])/2
    o[:,156], o[:,89] = direction, -direction
    c["z_post.0.weight"] = like258(w, c["z_post.0.weight"])
    c["z_post.0.bias"] = like258(b, c["z_post.0.bias"])
    c["z_post.2.weight"] = like258(o, c["z_post.2.weight"])
    return c

screen286 = {}
restore286 = {k:globals()[k] for k in factual_eval194.__code__.co_names
              if k in globals() and isinstance(globals()[k], dict)}
settings286 = [(2.25,.75,27.,.5), (2.25,.75,28.,.5),
               (2.,1.,28.,.5), (1.5,1.5,26.,.5),
               (2.25,.75,28.,.7)]
try:
    for setting in settings286:
        result = factual_eval194(candidate286(*setting))
        nll = float(np.asarray(a262(result[1])).mean())
        screen286[setting] = {"nll":nll, "delta":nll-1.6355364322662354}
finally:
    globals().update(restore286)
print("CPU SCREEN", screen286)

acceptable286 = [s for s in settings286 if screen286[s]["delta"] <= .0044]
chosen286 = next((s for s in settings286 if s in acceptable286), None)
backup286 = {n:torch.as_tensor(load(n)).clone() for n in edited()}
if chosen286 is not None:
    selected286 = candidate286(*chosen286)
    for n in ("z_post.0.weight","z_post.0.bias","z_post.2.weight"):
        save_edit(n, selected286[n])
    print("PROVISIONALLY SAVED", chosen286)
else:
    print("NO CANDIDATE SAVED")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 286\nRefit NE identity gate over empirical training dose ranges extended to half/double, including mixed bundles. CPU screening: "+repr(screen286)+". Provisionally selected "+repr(chosen286)+"; prior edit backed up in backup286. GPU rollout/factual validation pending. This is a transmission calibration, not new causal evidence.\n")
```
