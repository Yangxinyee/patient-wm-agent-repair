The activation search also failed to find a close match, and the final summary print hit a data-shape error. I’ll check the helper’s actual instructions and compare fresh inputs with stored inputs. This will distinguish an assembly bug from a mistaken encoding assumption.

```python
import dis
print("=== GENERAL ATOM HELPER INSTRUCTIONS ===")
print("; ".join(
    f"{x.opname} {x.argrepr}" for x in dis.get_instructions(recon_record_atom)
    if x.opname not in {"RESUME", "PRECALL", "CACHE", "PUSH_NULL"}
)[:6000])
print("\n=== SINGLETON CONSISTENCY ===")
for code in ["insulin", "norepinephrine", "glucose_finger", "map_arterial"]:
    if code in pure_action_rows:
        print(code, "pure_action_rows structure:",
              type(pure_action_rows[code]).__name__,
              [getattr(x, "shape", (len(x) if hasattr(x, "__len__") else None))
               for x in pure_action_rows[code]])
    row_indices = []
    for i in recon_valid_rows:
        if len(recon_obs_lists[i])+len(recon_action_lists[i]) != 1:
            continue
        rr0 = state_df.iloc[i]
        sub0 = tune[(tune.stay_id == rr0.stay_id) &
                    (np.abs(tune.hour-rr0.hour) < 1e-5)]
        if len(sub0) == 1 and sub0.iloc[0].event == code:
            row_indices.append(int(i))
        if len(row_indices) == 20:
            break
    ii0 = np.array(row_indices, dtype=int)
    stored_atoms = torch.stack([
        (recon_action_lists[i] if code in DRUGS else recon_obs_lists[i])[0]
        for i in ii0
    ])
    fresh_atoms = torch.stack([
        recon_record_atom(next(tune[(tune.stay_id == state_df.iloc[i].stay_id) &
                                   (np.abs(tune.hour-state_df.iloc[i].hour) < 1e-5)]
                               .itertuples(index=False)))
        for i in ii0
    ])
    fresh_ctx = recon_ctx(ii0, gap_style="log")
    print(code, "n", len(ii0),
          "atom max difference", float((fresh_atoms-stored_atoms).abs().max()),
          "context max difference", float((fresh_ctx-recon_context[ii0]).abs().max()))
    enc0 = single_encode_batch(fresh_atoms, "encode_action" if code in DRUGS else "encode")
    e0 = empty_e.repeat(len(ii0), 1) if code in DRUGS else enc0
    u0 = enc0 if code in DRUGS else torch.zeros_like(enc0)
    prediction0 = recon_mlp(torch.cat([fresh_ctx, e0, u0], dim=1), POST_PREFIX)[:, :32]
    print("posterior RMSE:",
          float(((prediction0-torch.as_tensor(S0["x"][ii0, 960:992]))**2).mean().sqrt()))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 26: assembly implementation audit\n"
            "Activation and unit searches did not solve observation mismatch. "
            "Comparing fresh versus retained atoms/contexts and singleton predictions. "
            "No edits. Python runs: 26; forward passes: 3.\n")
```
