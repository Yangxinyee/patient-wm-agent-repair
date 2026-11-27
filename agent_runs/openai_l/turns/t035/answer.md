For readings without a prior value, the error is now close to the estimated four-draw sampling noise, though not conclusively within it. The best settings use different activations for the prior and value head. I’ll apply those settings to later readings and test bin boundaries and register layouts. A consistent match across both glucose and MAP would support using the reconstruction for an edit.

```python
r31_edge_info = {}
for r31_event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    r31_spec = normspec22[(r31_event, "value")]
    r31_edge_info[r31_event] = np.asarray(r31_spec["fine_bin_edges"], dtype=float)
print("Fine-bin edges:", {k: v.tolist() for k, v in r31_edge_info.items()})

r31_bins = {}
r31_obsbins = {}
for r31_side in ("left", "right"):
    r31_b = np.zeros(len(ds22), dtype=np.int64)
    r31_ob = np.zeros(len(ds22), dtype=np.int64)
    for r31_event, r31_edges in r31_edge_info.items():
        r31_mask = ds22.event.to_numpy() == r31_event
        r31_present = r31_mask & (has22 > 0)
        r31_b[r31_present] = np.clip(
            np.searchsorted(r31_edges, ds22.last_value22.to_numpy()[r31_present],
                            side=r31_side)+1, 1, 16)
        r31_ob[r31_mask] = np.clip(
            np.searchsorted(r31_edges, ds22.observed.to_numpy()[r31_mask],
                            side=r31_side)+1, 1, 16)
    r31_bins[r31_side] = torch.tensor(r31_b)
    r31_obsbins[r31_side] = torch.tensor(r31_ob)
    print(r31_side, "last-bin differences from retained:",
          int((r31_bins[r31_side] != last_bin22).sum()),
          "observed-bin differences:",
          int((r31_obsbins[r31_side] != observed_bin22).sum()))

r31_configs = []
for r31_result in r30_results:
    r31_cfg = r31_result[2:9]
    if r31_cfg not in r31_configs:
        r31_configs.append(r31_cfg)
    if len(r31_configs) == 6:
        break
r31_classic = ("gelu", "seconds", .1, .1, "gelu", "gelu", 1)
if r31_classic not in r31_configs:
    r31_configs.append(r31_classic)

r31_screen = np.arange(min(450, len(ds22)))
r31_pre = torch.tensor(r29_export["x"][pre22[r31_screen]])
r31_dt = torch.tensor(dt22[r31_screen], dtype=torch.float32)
r31_base = r31_pre[:, :192] + r30_norm(
    (r31_pre[:, 192:960].reshape(-1, 4, 192) *
     torch.exp(-r31_dt[:, None, None]*torch.exp(r30_w("core.flow.a"))[None, None, :])).mean(1),
    "core.out_norm")
r31_code = code22[r31_screen]
r31_has = torch.tensor(has22[r31_screen], dtype=torch.float32)[:, None]
r31_z = torch.tensor(last_z22[r31_screen], dtype=torch.float32)[:, None]
r31_centres = centres22[r31_screen]
r31_ref = torch.tensor(reference_mean22[r31_screen], dtype=torch.float32)
r31_M = r30_w("attr_heads.value_bin.0.weight") @ r30_w("last_numeric.weight")
r31_best = []
r31_start = time.time()

for r31_cfg in r31_configs:
    r31_ga, r31_unit, r31_ps, r31_vs, r31_pa, r31_ha, r31_ti = r31_cfg
    r31_ld = torch.log1p(r31_dt/(3600 if r31_unit == "hours" else 1))
    r31_raw = r30_mlp(r31_base+r30_mlp(
        (r31_ld*r31_ps)[:, None], "gap", r31_ga), "z_prior", r31_pa)
    r31_q = r30_norm(
        r31_base+r30_mlp((r31_ld*r31_vs)[:, None], "gap", r31_ga)
        +r30_lin(r31_raw[:, :32], "z_head")+r30_w("type_emb.weight")[r31_ti],
        "query_norm")
    r31_qfirst = r30_lin(r31_q, "attr_heads.value_bin.0")
    r31_local = []
    for r31_side, r31_au in itertools.product(("left", "right"), ("seconds", "hours")):
        r31_lb = r31_bins[r31_side][r31_screen]
        r31_trans = r30_w("embed_transition.value_bin.weight")[r31_code*17+r31_lb]
        r31_const = (r30_w("embed.code.weight")[r31_code]
                     +r30_w("embed.bins.value.weight")[r31_lb]
                     +r30_w("embed.bins.rate.weight")[0]
                     +r30_w("last_numeric.bias")
                     +r30_w("last_status.weight")[0])
        r31_fixed = r31_qfirst+torch.nn.functional.linear(
            r31_const, r30_w("attr_heads.value_bin.0.weight"))
        r31_age = torch.tensor(
            np.log1p(age_hours22[r31_screen]*(3600 if r31_au == "seconds" else 1))/10,
            dtype=torch.float32)[:, None]
        for r31_zp, r31_ap in itertools.permutations(range(8), 2):
            r31_available = [i for i in range(8) if i not in (r31_zp, r31_ap)]
            r31_flags = [()] + [(i,) for i in r31_available] + list(
                itertools.combinations(r31_available, 2))
            r31_f0 = r31_fixed+r31_z*r31_M[:, r31_zp]+r31_age*r31_M[:, r31_ap]
            for r31_fl in r31_flags:
                r31_f = r31_f0
                if r31_fl:
                    r31_f = r31_f+r31_has*r31_M[:, list(r31_fl)].sum(1)
                r31_p = (r30_lin(r30_act(r31_f, r31_ha),
                                 "attr_heads.value_bin.2")+r31_trans).softmax(-1)
                r31_mean = (r31_p[:, 1:]*r31_centres).sum(1)
                r31_err = float((r31_mean-r31_ref).square().mean().sqrt())
                r31_local.append((r31_err, r31_cfg, r31_side, r31_au,
                                  r31_zp, r31_ap, r31_fl))
    r31_local.sort()
    r31_best.extend(r31_local[:5])

r31_best.sort()
print("Search seconds:", round(time.time()-r31_start, 1))
print("Best specifications:")
for r31_result in r31_best[:12]:
    print(r31_result)

with open("/work/notes.md", "a") as r31_note:
    r31_note.write(
        "\n## Run 31\n"
        "No-last head search favors GELU gap/prior and SiLU value head; first-reading "
        "MC RMSE 3.16 versus expected noise 2.71, not yet a definitive match. "
        "Testing independent fine-bin boundary conventions and eight-column last "
        "register mappings on later readings. Retained r31_bins/r31_obsbins, "
        "r31_configs, and r31_best. No edits.\n"
    )
```