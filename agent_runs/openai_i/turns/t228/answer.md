The neutral transplant was installed, but the output-file lookup failed afterward. Its innovation routing also showed a scaling error, so it is not ready for GPU validation. I’ll remove it, correct the unit-vector scaling, and verify the channel numerically before reinstalling. Near-unit innovation decoding with negligible inactive leakage would confirm the correction.

```python
for k in list(edited()):
    drop_edit(k)

anchor_unit204 = anchor196 / np.linalg.norm(anchor196)
ins_unit204 = axis_ins196 / np.linalg.norm(axis_ins196)
ne_unit204 = axis_ne196 / np.linalg.norm(axis_ne196)
print("Basis norms:", np.linalg.norm(anchor196), np.linalg.norm(axis_ins196),
      np.linalg.norm(axis_ne196))
print("Basis means:", np.mean(anchor_unit204), np.mean(ins_unit204), np.mean(ne_unit204))

zw204 = zin_newW196.copy()
zb204 = fit_zin196[-1].astype(np.float32) + 100.*np.sqrt(192.)*anchor_unit204
tokens204 = old_tokens196.clone()
tokens204[:, 6] = torch.tensor((Z201 @ zw204.T + zb204)[source201], dtype=torch.float32)

raw204 = attention_features196(tokens204, W["core.prelude.0.attn.q.weight"],
                              W["core.prelude.0.attn.kv.weight"])
oldstyle_features204 = array196(raw204[0] if isinstance(raw204, tuple) else raw204)
X204 = np.column_stack([
    oldstyle_features204.reshape(-1, 192)[:, keep_att201], np.ones(len(target_att201))])
xf204 = X204[fit_att201]
cov204 = xf204.T @ xf204
reg204 = np.eye(len(cov204)) * (1e-5*np.trace(cov204)/len(cov204))
reg204[-1, -1] = 0
coef204 = np.linalg.solve(cov204+reg204, xf204.T @ target_att201[fit_att201])

q204 = array196(load_original("core.prelude.0.attn.q.weight")).copy()
kv204 = array196(load_original("core.prelude.0.attn.kv.weight")).copy()
q204[head_slice203] = 0
kv204[head_slice203] = 0
kv204[192+head_start203:192+head_start203+48] = 0
q204[head_start203] = (np.sqrt(48.)*constant203).astype(np.float32)
kv204[head_start203] = (50.*anchor_unit204/(np.sqrt(192.)*gamma203)).astype(np.float32)
kv204[192+head_start203] = constant203.astype(np.float32)
for j, unit_axis in enumerate((ins_unit204, ne_unit204)):
    channel_scale = np.linalg.norm(zw204[:, private_dims196[j]])
    vr = (100./channel_scale) * unit_axis / gamma203
    vr -= float(vr @ beta203) * constant203
    kv204[192+head_start203+1+j] = vr.astype(np.float32)

out204 = np.zeros((192, 192), dtype=np.float32)
out204[:, keep_att201] = coef204[:-1].T.astype(np.float32)
out204[:, head_start203] = coef204[-1].astype(np.float32)

raw204 = attention_features196(tokens204, torch.tensor(q204), torch.tensor(kv204))
routed_features204 = raw204[0] if isinstance(raw204, tuple) else raw204
channels204 = array196(routed_features204)[:, :, head_start203+1:head_start203+3]
error204 = channels204-F195[source201, None, :]
print("Corrected innovation RMS:", np.sqrt(np.mean(error204**2, axis=(0, 1))))
print("Corrected innovation max errors:", np.max(np.abs(error204), axis=(0, 1)))
reconstruction204 = array196(torch.nn.functional.linear(
    routed_features204, torch.tensor(out204))).reshape(-1, 192)
print("Neutral held attention RMS:",
      np.sqrt(np.mean((reconstruction204[~fit_att201]-target_att201[~fit_att201])**2)))
assert np.max(np.abs(error204)) < .02, "Channel routing is not yet reliable."

neutral204 = {k: v.clone() for k, v in private_post201.items()}
neutral204.update({
    "z_in.weight": torch.tensor(zw204),
    "z_in.bias": torch.tensor(zb204, dtype=torch.float32),
    "core.prelude.0.attn.q.weight": torch.tensor(q204),
    "core.prelude.0.attn.kv.weight": torch.tensor(kv204),
    "core.prelude.0.attn.out.weight": torch.tensor(out204)
})
np.savez("/work/neutral204.npz", **{k: array196(v) for k, v in neutral204.items()})
for k, v in neutral204.items():
    save_edit(k, v)

tuning_columns204 = list(pd.read_csv("/data/tuning.csv", nrows=1).columns)
stay_col204 = next(c for c in ("stay_id", "stay") if c in tuning_columns204)
validation_stays204 = sorted(pd.read_csv(
    "/data/tuning.csv", usecols=[stay_col204])[stay_col204].unique().tolist())[:100]
print("Validation stays:", validation_stays204)
print("Active edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 203–204 — correct private routing geometry\n"
            "- Run 203 installed a zero-effect candidate but failed on an output-file "
            "lookup. The innovation readout exposed a unit-vector scaling error.\n"
            "- Removed that candidate, corrected anchor/key/value scaling, "
            "and numerically checked innovation transport before reinstalling.\n"
            "- Neutral204 is saved; drug-response columns are still zero. "
            "Next decision requires actual factual NLL on 100 tuning stays.\n")
```
