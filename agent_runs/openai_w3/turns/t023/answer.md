Bundle alignment is exact, and the two encoder implementations agree. The remaining mismatch could still come from timing or from how readings enter the posterior. I’ll test those separately; a correct reconstruction must explain measurements and drug starts with the same settings, not just fit one event type.

```python
from types import SimpleNamespace

vp = prev_idx[valid_idx]
old_out = sx[vp, :192]
old_h = sx[vp, 192:960].reshape(-1, 4, 192)
print("previous-index disagreements:", int(np.sum(vp != valid_idx-1)))
print("flow rates:", np.quantile(np.exp(as_numpy(W["core.flow.a"])), [0, .25, .5, .75, 1]))

exact_start_mask = np.array([
    len(bundle_atom_info[idx]) == 1 and not bundle_atom_info[idx][0][1]
    and bundle_atom_info[idx][0][0].event in ("insulin", "norepinephrine")
    for idx in valid_idx])
reading_mask = np.array([
    all(not ending0 and rec0.event.startswith(("glucose_", "map_"))
        for rec0, ending0 in bundle_atom_info[idx])
    for idx in valid_idx])

print("=== FLOW/GAP CHECK ===")
with torch.no_grad():
    gap_options = {
        "log1p_seconds": torch.log1p(dt_all).reshape(-1, 1)/10,
        "log_seconds": torch.log(dt_all.clamp_min(1)).reshape(-1, 1)/10,
        "log1p_hours": torch.log1p(dt_all/3600).reshape(-1, 1)/10,
        "log1p_minutes": torch.log1p(dt_all/60).reshape(-1, 1)/10,
    }
    timing_tests = []
    for flow_scale in (0., 1/3600, 1/60, .1, .5, 1., 2.):
        hf0 = old_h * torch.exp(-dt_all[:, None, None] * flow_scale
                                * torch.exp(W["core.flow.a"])[None, None, :])
        base0 = old_out + layer_norm(hf0.mean(1), "core.out_norm")
        for gap_name, gap_input in gap_options.items():
            cx0 = base0 + mlp(gap_input, "gap", "gelu")
            zz0 = mlp(torch.cat([cx0, e_all[valid_idx], u_all[valid_idx]], 1),
                      "z_post", "gelu")[:, :32]
            er0 = ((zz0 - sx[valid_idx, -32:])**2).mean(1).numpy()
            timing_tests.append((float(er0[reading_mask].mean()),
                                 float(er0[exact_start_mask].mean()),
                                 flow_scale, gap_name,
                                 float((cx0-ctx_all).abs().max())))
    for result0 in sorted(timing_tests)[:10]:
        print("reading MSE, drug-start MSE, flow scale, gap, context difference:", result0)

    print("=== READING ATTRIBUTE CHECK ===")
    for event0 in ("map_cuff", "glucose_finger"):
        idx0 = np.array(isolated_by_event[event0][:32])
        p0 = np.array([pos_by_idx[int(i0)] for i0 in idx0])
        recs0 = [bundle_atom_info[i0][0][0] for i0 in idx0]
        atom0 = torch.stack([make_atom(r0, False, "preserve") for r0 in recs0])
        missing0 = torch.stack([
            make_atom(SimpleNamespace(**{**r0._asdict(), "value": np.nan}),
                      False, "preserve") for r0 in recs0])
        for mode0, aa0 in (("full", atom0), ("no_attributes", missing0)):
            for encoder0 in ("encode", "encode_action"):
                ee0 = encode_one_set(aa0, encoder0)
                obs0 = ee0 if encoder0 == "encode" else empty_obs.expand(len(idx0), -1)
                act0 = ee0 if encoder0 == "encode_action" else torch.zeros_like(ee0)
                zz0 = mlp(torch.cat([ctx_all[p0], obs0, act0], 1),
                          "z_post", "gelu")[:, :32]
                print(event0, mode0, encoder0,
                      float(((zz0-sx[idx0, -32:])**2).mean()))
```