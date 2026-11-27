The norepinephrine decoders transmit dose changes accurately, but predict factual innovations noisily. That makes factual-fit testing essential. I’ll build an unsaved second channel and test whether both signals remain recoverable; poor held-out decoding or disruption of the insulin channel would reject it before submission.

```python
from scipy.linalg import null_space

# Preserve the submitted model. This run tests a candidate only in CPU memory.
names186 = [
    "z_post.0.weight", "z_post.0.bias", "z_post.2.weight", "z_post.2.bias",
    "z_in.weight", "core.prelude.0.attn.kv.weight",
    "core.prelude.0.attn.out.weight"
]
cpu_backup186 = {n: rW[n].clone() for n in names186}
dual_channel186 = {n: as_np184(load(n)).copy() for n in names186}

axis_ne186 = null_space(as_np184(latent_axis104).reshape(1,-1))[:,0]
gamma186 = as_np184(gamma104)
keyrow186 = dual_channel186["core.prelude.0.attn.kv.weight"][48]
constraints186 = np.stack([
    1/gamma186, as_np184(signature104)/gamma186,
    as_np184(carrier104)/gamma186, as_np184(bias104),
    gamma186*as_np184(signature104),
    gamma186*as_np184(carrier104), keyrow186
])
basis186 = null_space(constraints186)
cov186 = basis186.T @ as_np184(S104) @ basis186
ev186, vec186 = np.linalg.eigh((cov186+cov186.T)/2)
vunit186 = basis186 @ vec186[:,0]
carrier_ne186 = vunit186/gamma186
carrier_ne186 /= np.linalg.norm(carrier_ne186)
value_ne186 = vunit186 / (
    (vunit186 @ (gamma186*carrier_ne186))*lambda104*np.sqrt(192)
)

out186 = dual_channel186["core.prelude.0.attn.out.weight"]
load_basis186 = null_space(np.stack([out186[:,48],out186[:,49],np.ones(192)]))
loading_ne186 = load_basis186 @ (load_basis186.T @ carrier_ne186)
loading_ne186 /= np.linalg.norm(loading_ne186)

coef_ne186 = raw_candidates87[("norepinephrine", .01, 10.0)].copy()
gate_ne186 = np.zeros(576)
gate_ne186[384:] = gate_fits180[100.0].coef_[0]*40
gate_bias186 = float(gate_fits180[100.0].intercept_[0])*40
for row, sign in ((156,1),(89,-1)):
    dual_channel186["z_post.0.weight"][row] = gate_ne186 + sign*coef_ne186[:576]/2
    dual_channel186["z_post.0.bias"][row] = gate_bias186 + sign*coef_ne186[576]/2

project_ne186 = np.eye(32)-np.outer(axis_ne186,axis_ne186)
dual_channel186["z_post.2.weight"][:32] = (
    project_ne186 @ dual_channel186["z_post.2.weight"][:32]
)
dual_channel186["z_post.2.bias"][:32] = (
    project_ne186 @ dual_channel186["z_post.2.bias"][:32]
)
dual_channel186["z_post.2.weight"][:32,156] += axis_ne186
dual_channel186["z_post.2.weight"][:32,89] -= axis_ne186
dual_channel186["z_in.weight"] += (
    np.sqrt(192)*100*lambda104*np.outer(carrier_ne186,axis_ne186)
).astype(np.float32)
dual_channel186["core.prelude.0.attn.kv.weight"][192+50] = value_ne186
dual_channel186["core.prelude.0.attn.out.weight"][:,50] = 80*loading_ne186

def collect_channel186():
    es, zs = [], []
    for start in range(0,len(indices102),256):
        _, cap = capture_transition102(indices102[start:start+256])
        es.append(as_np184(cap["ep"]))
        zs.append(as_np184(cap["zk"]))
    return np.concatenate(es), np.concatenate(zs)

try:
    Eold186, Zold186 = collect_channel186()
    for n,a in dual_channel186.items():
        rW[n] = torch.from_numpy(a.copy()).float()
    Enew186, Znew186 = collect_channel186()
finally:
    for n,a in cpu_backup186.items():
        rW[n] = a

q_ne186 = Znew186 @ axis_ne186
q_ins186 = Znew186 @ as_np184(latent_axis104)
old_ins_signal186 = np.einsum("nsd,d->ns", Eold186, as_np184(connection_decoder123))
target_ne186 = np.repeat(q_ne186,4)
target_ins186 = old_ins_signal186.reshape(-1)
features186 = Enew186.reshape(-1,192).astype(np.float64)
trslots186 = np.repeat(np.asarray(train180,bool),4)
teslots186 = np.repeat(np.asarray(test180,bool),4)
gram186 = features186[trslots186].T @ features186[trslots186]
rhs_ne186 = features186[trslots186].T @ target_ne186[trslots186]
rhs_ins186 = features186[trslots186].T @ target_ins186[trslots186]
decoders_ne186, decoders_ins186 = {}, {}
for ridge in (.0001,.01,.1,1.0):
    dn = np.linalg.solve(gram186+ridge*np.eye(192),rhs_ne186)
    di = np.linalg.solve(gram186+ridge*np.eye(192),rhs_ins186)
    decoders_ne186[ridge], decoders_ins186[ridge] = dn,di
    pn, pi = features186 @ dn, features186 @ di
    ne_active = teslots186 & np.repeat(np.asarray(active_ne180,bool),4)
    ne_inactive = teslots186 & ~np.repeat(np.asarray(active_ne180,bool),4)
    print("RIDGE",ridge,
          "NE R2 all/active",
          round(float(r2_185(target_ne186[teslots186],pn[teslots186])),5),
          round(float(r2_185(target_ne186[ne_active],pn[ne_active])),5),
          "NE inactive RMS",round(float(np.sqrt(np.mean(pn[ne_inactive]**2))),6),
          "insulin preservation RMSE",
          round(float(np.sqrt(np.mean((pi[teslots186]-target_ins186[teslots186])**2))),6))

print("posterior insulin preservation RMS",
      np.sqrt(np.mean((q_ins186-Zold186@as_np184(latent_axis104))**2)))
print("NE posterior vs intended gated decoder RMS",
      np.sqrt(np.mean((q_ne186-gated_ne185(raw87,coef_ne186))**2)))
print("NE carrier/old carrier dot",carrier_ne186@as_np184(carrier104))
print("NE key sensitivity",keyrow186@(gamma186*carrier_ne186))
print("candidate tensors",list(dual_channel186))

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Run 186\n"
        "Run-185 NE decoders had near-unit imposed-dose slopes but poor "
        "held-out innovation R2. Constructed an UNSAVED dual channel using "
        "posterior rows 156/89, a second latent carrier and reserved-head value "
        "dimension 50; tested held-out prelude decoding and insulin preservation. "
        "CPU weights restored in finally; submitted edits unchanged. Factual "
        "fit and free rollouts, not channel decoding, will determine acceptance.\n"
    )
```