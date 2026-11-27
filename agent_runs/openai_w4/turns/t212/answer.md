The prototype preserves the insulin posterior signal, but its norepinephrine channel is only about 92% linearly decodable. I’ll try smaller amplitudes and include half-dose and double-dose cases in the decoder fit. Accurate held-out dose increments with preserved insulin responses would support proceeding to factual testing.

```python
torch.set_num_threads(8)

delta_b187 = as_np184(load("core.b_in.weight")) - as_np184(load_original("core.b_in.weight"))
old_decoder187 = as_np184(connection_decoder123).astype(np.float64)
ins_direction187 = delta_b187 @ old_decoder187 / (old_decoder187 @ old_decoder187)
print("existing B edit rank-one relative error",
      np.linalg.norm(delta_b187-np.outer(ins_direction187,old_decoder187))
      / np.linalg.norm(delta_b187))

target_ne187 = np.where(
    np.isin(probe_events118,["map_arterial","map_cuff"]), 3.1429, 0.0
)
J187 = as_np184(JJ118).astype(np.float64)
directions_ne187 = {}
for ridge in (.001,.01,.1,1.0):
    v = np.linalg.solve(J187.T@J187+ridge*np.eye(192),J187.T@target_ne187)
    directions_ne187[ridge] = v
    pred = J187@v
    print("NE response ridge",ridge,"norm",round(float(np.linalg.norm(v)),4),
          "event means",[(e,round(float(pred[probe_events118==e].mean()),4))
                         for e in np.unique(probe_events118)])

def collect_selected187(ix, actions=None):
    es, zs = [], []
    for start in range(0,len(ix),256):
        stop = start+256
        override = None if actions is None else torch.from_numpy(
            np.asarray(actions[start:stop],dtype=np.float32).copy()
        )
        _, cap = capture_transition102(ix[start:stop],action_override=override)
        es.append(as_np184(cap["ep"]))
        zs.append(as_np184(cap["zk"]))
    return np.concatenate(es),np.concatenate(zs)

# Verify override indexing before touching the CPU weight dictionary.
trial_u187 = np.asarray(raw_cf87[("norepinephrine",2.0)][:8,384:],dtype=np.float32)
_, trial_cap187 = capture_transition102(
    indices102[:8],action_override=torch.from_numpy(trial_u187.copy())
)
assert np.allclose(as_np184(trial_cap187["uk"]),trial_u187,atol=1e-6)
print("action override parity verified")

active_pos187 = np.flatnonzero(active_ne180)
active_ix187 = indices102[active_pos187]
cf_actions187 = {
    f: np.asarray(raw_cf87[("norepinephrine",f)][active_pos187,384:],dtype=np.float32)
    for f in (.5,2.0)
}
old_cf187 = {
    f: collect_selected187(active_ix187,cf_actions187[f])[0] for f in (.5,2.0)
}
old_targets187 = np.concatenate([
    old_ins_signal186,
    np.einsum("nsd,d->ns",old_cf187[.5],old_decoder187),
    np.einsum("nsd,d->ns",old_cf187[2.0],old_decoder187)
]).reshape(-1)
train_bundles187 = np.concatenate([
    train180,train180[active_pos187],train180[active_pos187]
]).astype(bool)
test_bundles187 = ~train_bundles187
tr187 = np.repeat(train_bundles187,4)
te187 = np.repeat(test_bundles187,4)
amplitude_tests187 = {}

try:
    for n,a in dual_channel186.items():
        rW[n] = torch.from_numpy(a.copy()).float()
    for amplitude in (10.0,20.0,40.0):
        out = dual_channel186["core.prelude.0.attn.out.weight"].copy()
        out[:,50] = amplitude*loading_ne186
        rW["core.prelude.0.attn.out.weight"] = torch.from_numpy(out)
        eb,zb = collect_selected187(indices102)
        cases = {f: collect_selected187(active_ix187,cf_actions187[f]) for f in (.5,2.0)}
        all_e = np.concatenate([eb,cases[.5][0],cases[2.0][0]])
        all_q = np.concatenate([
            zb@axis_ne186,cases[.5][1]@axis_ne186,cases[2.0][1]@axis_ne186
        ])
        a = all_e.reshape(-1,192).astype(np.float64)
        y = np.stack([np.repeat(all_q,4),old_targets187],axis=1)
        gram = a[tr187].T@a[tr187]
        rhs = a[tr187].T@y[tr187]
        fitted = np.linalg.solve(gram+.0001*np.eye(192),rhs)
        p = a@fitted
        decoded = p[:,0].reshape(-1,4).mean(axis=1)
        base_decoded = decoded[:len(indices102)]
        summaries = []
        offset = len(indices102)
        held_active = test180[active_pos187].astype(bool)
        for f in (.5,2.0):
            cf_decoded = decoded[offset:offset+len(active_pos187)]
            expected = (f-1)*delta185[active_pos187]
            use = held_active & (abs(expected)>1e-8)
            change = cf_decoded-base_decoded[active_pos187]
            slope = np.dot(expected[use],change[use])/np.dot(expected[use],expected[use])
            summaries.append((f,round(float(slope),5),
                              round(float(np.sqrt(np.mean((change[use]-expected[use])**2))),5)))
            offset += len(active_pos187)
        amplitude_tests187[amplitude] = {
            "decoder_ne":fitted[:,0], "decoder_ins":fitted[:,1],
            "Ebase":eb, "out_weight":out, "dose_tests":summaries
        }
        print("AMPLITUDE",amplitude,
              "NE held R2",round(float(r2_185(y[te187,0],p[te187,0])),5),
              "insulin preservation RMSE",
              round(float(np.sqrt(np.mean((p[te187,1]-y[te187,1])**2))),6),
              "dose slope/RMSE",summaries,
              "decoder norms",np.linalg.norm(fitted,axis=0).round(3))
finally:
    for n,a in cpu_backup186.items():
        rW[n] = a

print("\nFACTUAL REPLAY INTERFACES")
for name,obj in list(globals().items()):
    if callable(obj) and hasattr(obj,"__code__") and (
        name.startswith("replay") or name in ("probe_means95","transition_with_b110")
    ):
        print(name,inspect.signature(obj),"dependencies",obj.__code__.co_names[:25])

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Run 187\n"
        "Run-186 preserved insulin posterior signal (RMS 1.5e-6) but NE "
        "prelude decoding R2 was .92 at amplitude 80. Tested amplitudes "
        "10/20/40 with factual and half/double NE dose encodings, jointly "
        "fitting NE decoding and preservation of the current insulin B_in "
        "signal. Also computed MAP-selective response directions from the "
        "retained finite-difference Jacobian. CPU weights restored; no "
        "submitted weights changed. Record effect estimates remain conditional "
        "on the identification assumptions, not certified by these tests.\n"
    )
```