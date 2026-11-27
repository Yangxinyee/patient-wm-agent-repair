On these stays, the record-kernel estimates are insulin −15.01/+7.50 and norepinephrine +4.87/−2.43 when weighted like the tool’s summary. The current edit underresponds to doubling both drugs. I’ll test a norepinephrine identity gate over a wider dose range, then screen it for factual safety. Reliable activation across doses and a small NLL penalty would justify a rollout test.

```python
import torch
from sklearn.svm import LinearSVC

base285 = {n: torch.as_tensor(load(n)).clone() for n in NAMES}
w285 = a262(base285["z_post.0.weight"]).copy()
b285 = a262(base285["z_post.0.bias"]).copy()
o285 = a262(base285["z_post.2.weight"]).copy()
qw285 = w285[156] - w285[89]
qb285 = float(b285[156] - b285[89])
print("CURRENT NE GATE BLOCK NORMS",
      [float(np.linalg.norm(w285[89, a:b])) for a,b in ((0,192),(192,384),(384,576))],
      "bias", float(b285[89]), "innovation norm/bias", float(np.linalg.norm(qw285)), qb285)

rates_ne285 = np.unique(np.r_[0.0, np.geomspace(0.001, 4.0, 100)])
rates_ins285 = np.unique(np.r_[0.0, np.geomspace(0.001, 32.0, 100)])
atoms_ne285 = atom_exact65("norepinephrine", np.full(len(rates_ne285), np.nan), rates_ne285, rates_ne285)
atoms_ins285 = atom_exact65("insulin", np.full(len(rates_ins285), np.nan), rates_ins285, rates_ins285)
u_ne285 = singleton_action_repair(atoms_ne285)
u_ins285 = singleton_action_repair(atoms_ins285)

rng285 = np.random.default_rng(285)
ii285 = rng285.integers(0, len(rates_ins285), 160)
jj285 = rng285.integers(0, len(rates_ne285), 160)
mix_hidden285 = (
    rp_mlp(atoms_ins285[ii285], "encode_action.mlp1")
    + rp_mlp(atoms_ne285[jj285], "encode_action.mlp1")
) / np.sqrt(2.0)
mix_hidden285 = mix_hidden285 + rp_linear(
    torch.full((len(ii285), 1), float(np.log(3.0))), "encode_action.size"
)
u_mix285 = rp_norm(rp_mlp(mix_hidden285, "encode_action.mlp2"), "encode_action.norm")

xgate285 = np.vstack([a262(u_ne285), a262(u_mix285), a262(u_ins285), np.zeros((1,192))])
ygate285 = np.r_[np.ones(len(u_ne285)+len(u_mix285)),
                np.zeros(len(u_ins285)+1)]
gate_robust285 = LinearSVC(C=10.0, dual=False, max_iter=30000).fit(xgate285, ygate285)
for label, u in [("NE", u_ne285), ("mixed", u_mix285), ("insulin", u_ins285),
                 ("none", np.zeros((1,192)))]:
    new_margin = gate_robust285.decision_function(a262(u))
    old_margin = gate_models88["norepinephrine"].decision_function(a262(u))
    print(label, "old margin min/max", float(old_margin.min()), float(old_margin.max()),
          "new margin min/max", float(new_margin.min()), float(new_margin.max()))
print("NEW CLASSIFIER ERRORS", int(np.sum(gate_robust285.predict(xgate285) != ygate285)))

def robust_ne285(gain=1.0, gate_scale=40.0):
    c = {n: v.clone() for n,v in base285.items()}
    w, b, o = w285.copy(), b285.copy(), o285.copy()
    gw = np.zeros(576, dtype=np.float32)
    gw[384:] = gate_scale * gate_robust285.coef_[0]
    gb = float(gate_scale * gate_robust285.intercept_[0])
    w[156], w[89] = gw + gain*qw285/2, gw - gain*qw285/2
    b[156], b[89] = gb + gain*qb285/2, gb - gain*qb285/2
    direction = (o[:,156] - o[:,89])/2
    o[:,156], o[:,89] = direction, -direction
    c["z_post.0.weight"] = like258(w, c["z_post.0.weight"])
    c["z_post.0.bias"] = like258(b, c["z_post.0.bias"])
    c["z_post.2.weight"] = like258(o, c["z_post.2.weight"])
    return c

restore285 = {
    k: globals()[k] for k in factual_eval194.__code__.co_names
    if k in globals() and isinstance(globals()[k], dict)
}
screen285 = {}
try:
    for gain, scale in ((0.5,40.0),(0.8,40.0),(1.0,40.0),(1.0,80.0)):
        result = factual_eval194(robust_ne285(gain, scale))
        nll = float(np.asarray(a262(result[1])).mean())
        screen285[(gain,scale)] = {"nll":nll, "delta":nll-1.6355364322662354}
finally:
    globals().update(restore285)
print("UNSAVED FACTUAL SCREEN", screen285)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 285\nFitted a small NE identity classifier on exact action embeddings spanning rates 0–4, insulin 0–32, mixed actions and no action; no model gradients. Built unsaved shared-gate opposing NE pair and screened CPU factual loss: "+repr(screen285)+". Model dictionaries restored; submitted weights unchanged. Matched record targets (row means): insulin -15.009/+7.505, NE +4.866/-2.433; equal-stay targets -13.193/+6.597 and +4.705/-2.353.\n")
```
