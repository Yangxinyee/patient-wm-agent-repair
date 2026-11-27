The recovered caches show that the decoder used features with register directions removed. Applying those coefficients directly to raw head inputs could explain the failure. I’ll compare the rejected edit on both representations; large corrections only on raw inputs would confirm that projection was omitted from the installed weights.

```python
print("=== FEATURE REPRESENTATIONS ===")
feature_arrays = {}
for key, value in rec_features.items():
    aa = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    print(key, aa.shape)
    if aa.shape == rec_head_X.shape:
        feature_arrays[key] = aa
        print("  RMS difference from decoder input:",
              np.sqrt(np.mean((aa-rec_head_X)**2)))

print("\n=== PROJECTION CHECK ===")
Pcheck = np.asarray(trial_P, dtype=np.float64)
for name in ["trial_dv", "trial_av", "rec_gate_v"]:
    vcheck = np.asarray(globals()[name])
    print(name, "norm", np.linalg.norm(vcheck),
          "off-projector norm", np.linalg.norm(vcheck-Pcheck@vcheck))
for jcheck in replaced_rows.tolist():
    rowcheck = rW1[jcheck].numpy().astype(np.float64)
    print("installed row", jcheck, "norm", np.linalg.norm(rowcheck),
          "off-projector norm", np.linalg.norm(rowcheck-rowcheck@Pcheck))
print("Changed row biases:", rb1[replaced_rows].tolist())
print("Output changed-column norms:",
      torch.linalg.vector_norm(rW2[:, replaced_rows], dim=0).tolist())

print("\n=== LOCAL LOGIT DAMAGE ===")
candidate_inputs = {"decoder_input": rec_head_X}
candidate_inputs.update(feature_arrays)
for labelcheck, xcheck in candidate_inputs.items():
    txcheck = as_t(xcheck)
    newcheck = F.gelu(txcheck@rW1.T+rb1)@rW2.T+rb2
    oldcheck = F.gelu(txcheck@oW1.T+ob1)@oW2.T+ob2
    dc = (newcheck-oldcheck).numpy()
    print(labelcheck)
    for ec in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        mc = head_meta.event.to_numpy() == ec
        print(" ", ec, "RMS logits median/max",
              np.round(np.quantile(np.sqrt((dc[mc]**2).mean(1)), [.5,1]), 4).tolist())

print("\n=== RECORD TARGET SUMMARY ===")
print(json.dumps(brief(record_targets), indent=2)[:1800])
print("\n=== SELECTED TRIAL SETTINGS ===")
for name in ["trial_B", "trial_threshold", "trial_gate_strength",
             "trial_tilt_scale", "trial_db", "trial_ab", "trial_gb"]:
    print(name, globals().get(name))
print("\nCurrent edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 57: rejected-head projection audit\n"
            "Checking whether register-projected decoder features were mistakenly "
            "installed as coefficients on raw with-code inputs. No edits installed.\n")
```