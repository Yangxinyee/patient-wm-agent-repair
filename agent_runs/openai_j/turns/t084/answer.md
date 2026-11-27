The returned output repeats the reconstruction check; it does not yet show readout contrasts. I’ll compute those contrasts now, anchoring each changed state to its saved original state. This will show whether the weak rollout response begins with weak immediate dose sensitivity or instead develops during feedback.

```python
import torch.nn.functional as Fnn

def repair_changed_action(drug, factor):
    av = post_av0.clone()
    amask = ((repair_atoms.event.to_numpy() == drug)
             & ~repair_atoms.is_end.to_numpy())
    ix = np.flatnonzero(amask)
    for attr, column in [("dose", 2), ("rate", 4)]:
        candidates = [(drug, attr), (repair_codes[drug], attr),
                      (str(repair_codes[drug]), attr)]
        spec = next((repair_norms[k] for k in candidates if k in repair_norms), None)
        if spec is None:
            raise KeyError((drug, attr, list(repair_norms.keys())[:10]))
        old = repair_atoms.loc[amask, attr].to_numpy(float)
        new = old * factor
        old_z = np.clip((old-spec["median"])/spec["iqr"], -10, 10)
        new_z = np.clip((new-spec["median"])/spec["iqr"], -10, 10)
        edges = np.asarray(spec["fine_bin_edges"])
        old_bin = np.searchsorted(edges, old, side="left") + 1
        new_bin = np.searchsorted(edges, new, side="left") + 1
        delta = (wt(f"embed.bins.{attr}.weight")[new_bin]
                 - wt(f"embed.bins.{attr}.weight")[old_bin])
        delta += torch.tensor(new_z-old_z, dtype=torch.float32)[:, None] * \
                 wt("embed.numeric.weight")[:, column][None]
        av[ix] += delta
    return repair_set_encode(av, post_action_mask, "encode_action", zero_empty=True)

def repair_core_rows(rows, all_u, all_z, batch_size=96):
    out_parts, h_parts = [], []
    for begin in range(0, len(rows), batch_size):
        rr = rows[begin:begin+batch_size]
        dt = torch.tensor((STIME[rr]-STIME[rr-1])*3600, dtype=torch.float32)
        h = torch.tensor(SX[rr-1, 192:960]).reshape(-1, 4, 192)
        h *= torch.exp(-dt[:, None, None]*torch.exp(wt("core.flow.a")))
        mi = [np.flatnonzero((SSID == SSID[r]) &
                            (np.arange(len(SX)) <= r))[-128:] for r in rr]
        width = max(map(len, mi))
        memory = torch.zeros(len(rr), width, 192)
        ages = torch.zeros(len(rr), width)
        valid_memory = torch.zeros(len(rr), width, dtype=torch.bool)
        for j, indices in enumerate(mi):
            memory[j, :len(indices)] = post_e0[indices]
            ages[j, :len(indices)] = torch.tensor(
                (STIME[rr[j]]-STIME[indices])*3600, dtype=torch.float32)
            valid_memory[j, :len(indices)] = True
        oo, hh = repair_core_step(
            h, post_e0[rr], all_u[rr], all_z[rr],
            memory, ages, valid_memory, age_scale=1.0, age_mode="inputboth")
        out_parts.append(oo.numpy())
        h_parts.append(hh.numpy())
    return np.concatenate(out_parts), np.concatenate(h_parts)

def repair_centres_for_meta(meta):
    table = VOCAB["value_bin_centres"]
    result = []
    for code, event in zip(meta.code, meta.event):
        if isinstance(table, dict):
            keys = [str(int(code)), int(code), event]
            value = next((table[k] for k in keys if k in table), None)
            if value is None:
                raise KeyError((code, event, list(table.keys())[:8]))
        else:
            value = table[int(code)]
        result.append(value)
    return np.asarray(result, dtype=np.float32)

print("NORMALISATION KEYS:", list(repair_norms.keys())[:6])
readout_action_rows = np.flatnonzero(
    (post_rates.sum(1) > 0) & post_valid &
    np.r_[False, SSID[1:] == SSID[:-1]])
readout_base_out, readout_base_h = repair_core_rows(
    readout_action_rows, post_u0, post_z0[:, :32])
readout_truth_h = SX[readout_action_rows, 192:960].reshape(-1, 4, 192)
print("BASELINE:", len(readout_action_rows), "actions; out RMSE",
      float(np.sqrt(np.mean((readout_base_out-SX[readout_action_rows, :192])**2))),
      "h RMSE", float(np.sqrt(np.mean((readout_base_h-readout_truth_h)**2))),
      "h RMS", float(np.sqrt(np.mean(readout_truth_h**2))))

readout_base = reconstruct_reading_head(head_meta)
readout_centres = repair_centres_for_meta(head_meta)
readout_base_mu = (readout_base["probs"].numpy()*readout_centres).sum(1)
readout_cf_features, readout_cf_logits, readout_cf_states = {}, {}, {}
readout_contrast_rows = []
reading_state = head_meta.state_i.to_numpy(int)

for drug_index, drug in enumerate(["insulin", "norepinephrine"]):
    for factor in [0.5, 2.0]:
        uu = repair_changed_action(drug, factor)
        ph = Fnn.gelu(Fnn.linear(
            torch.cat([post_ctx0, post_e0, uu], 1),
            wt("z_post.0.weight"), wt("z_post.0.bias")))
        zz = Fnn.linear(ph, wt("z_post.2.weight"), wt("z_post.2.bias"))[:, :32]
        _, changed_h = repair_core_rows(readout_action_rows, uu, zz)
        state_cf = SX.copy()
        anchored = readout_truth_h + changed_h - readout_base_h
        state_cf[readout_action_rows, 192:960] = anchored.reshape(-1, 768)
        state_cf[readout_action_rows, :192] = ln(
            torch.tensor(anchored).mean(1), "core.out_norm").numpy()
        recovered = reconstruct_reading_head(head_meta, state_array=state_cf)
        readout_cf_states[(drug, factor)] = state_cf
        readout_cf_features[(drug, factor)] = recovered["hidden"].numpy()
        readout_cf_logits[(drug, factor)] = recovered["logits"].numpy()
        mu_change = (recovered["probs"].numpy()*readout_centres).sum(1)-readout_base_mu
        dose_change = (factor-1)*post_rates[reading_state, drug_index]
        for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
            mask = (head_meta.event.to_numpy() == event) & (dose_change != 0)
            if mask.any():
                readout_contrast_rows.append({
                    "drug": drug, "factor": factor, "sensor": event,
                    "n": int(mask.sum()),
                    "mean_shift": float(mu_change[mask].mean()),
                    "shift_per_rate": float(
                        dose_change[mask] @ mu_change[mask] /
                        (dose_change[mask] @ dose_change[mask]))
                })
print(pd.DataFrame(readout_contrast_rows).to_string(index=False))
print("ACTIVE EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Recurrent-to-readout contrast analysis\n"
            "- Reconstructed local action updates using current-inclusive bundle memory "
            "and unscaled log1p(age) memory embeddings.\n"
            "- Anchored counterfactual state differences to saved original states; "
            "measuring immediate value-head dose sensitivities before any weight change.\n")
```