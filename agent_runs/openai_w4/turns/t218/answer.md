Recalibration improved the held-out slopes to about 0.99 for half doses and 0.95 for double doses, at the cost of a modest factual-signal change. I’ll test that decoder in the two-drug channel and refit the insulin-preservation mapping. This is worthwhile only if dose transmission improves without disrupting the existing insulin response.

```python
chosen_ne193 = calibrations192[(0.001, 100.0)]["coef"].copy()
submitted193 = {
    name: torch.as_tensor(as_np184(load(name)).copy(), dtype=torch.float32)
    for name in NAMES
}
dual_candidate193 = {
    name: torch.as_tensor(as_np184(value).copy(), dtype=torch.float32)
    for name, value in dual_channel186.items()
}
dual_candidate193["core.prelude.0.attn.out.weight"] = torch.as_tensor(
    as_np184(amplitude_tests187[20.0]["out_weight"]).copy(),
    dtype=torch.float32
)

print("POSTERIOR PAIR CHECK")
for row in (156, 89):
    w = as_np184(dual_candidate193["z_post.0.weight"][row])
    b = float(dual_candidate193["z_post.0.bias"][row])
    errors = []
    for sign in (-1, 1):
        err = np.linalg.norm(
            w - gate_ne186 - sign * np.asarray(coef_ne186[:576]) / 2
        )
        berr = abs(b - gate_bias186 - sign * float(coef_ne186[576]) / 2)
        errors.append((float(err + berr), sign))
    error, sign = min(errors)
    print("row", row, "sign", sign, "reconstruction error", error)
    assert error < 0.01
    dual_candidate193["z_post.0.weight"][row] += torch.as_tensor(
        sign * (chosen_ne193[:576] - coef_ne186[:576]) / 2,
        dtype=torch.float32
    )
    dual_candidate193["z_post.0.bias"][row] += float(
        sign * (chosen_ne193[576] - coef_ne186[576]) / 2
    )

try:
    rW = dict(submitted193)
    rW.update(dual_candidate193)
    Ebase193, Zbase193 = collect_selected187(ix180)
    Ecf193, Zcf193 = {}, {}
    for factor in (0.5, 2.0):
        Ecf193[factor], Zcf193[factor] = collect_selected187(
            active_ix187, cf_actions187[factor]
        )
finally:
    rW = dict(submitted193)

features193 = np.concatenate([
    Ebase193, Ecf193[0.5], Ecf193[2.0]
]).reshape(-1, 192).astype(np.float64)

qbase193 = gated_ne185(raw_base192, chosen_ne193)
qcf193 = {
    factor: gated_ne185(raw_cf192[factor], chosen_ne193)[active_pos187]
    for factor in (0.5, 2.0)
}
target193 = np.column_stack([
    np.repeat(np.concatenate([qbase193, qcf193[0.5], qcf193[2.0]]), 4),
    np.asarray(old_targets187, dtype=np.float64)
])
train_b193 = np.concatenate([
    train192, train192[active_pos187], train192[active_pos187]
])
test_b193 = np.concatenate([
    test192, test192[active_pos187], test192[active_pos187]
])
trslots193 = np.repeat(train_b193, 4)
teslots193 = np.repeat(test_b193, 4)

gram193 = features193[trslots193].T @ features193[trslots193]
rhs193 = features193[trslots193].T @ target193[trslots193]
joint_decoders193 = {}

print("\nJOINT CHANNEL VALIDATION")
for ridge in (0.0001, 0.01, 0.1, 1.0):
    coefs = solve(gram193 + ridge*np.eye(192), rhs193, assume_a="pos")
    predictions = features193 @ coefs
    ne_r2 = r2_185(target193[teslots193, 0], predictions[teslots193, 0])
    ins_rmse = float(np.sqrt(np.mean(
        (predictions[teslots193, 1] - target193[teslots193, 1])**2
    )))
    base_prediction = (Ebase193 @ coefs[:, 0]).mean(axis=1)
    dose_results = []
    for factor in (0.5, 2.0):
        cf_prediction = (Ecf193[factor] @ coefs[:, 0]).mean(axis=1)
        change = cf_prediction - base_prediction[active_pos187]
        target = (factor - 1) * rate192[active_pos187]
        mask = test192[active_pos187]
        slope = float(np.dot(change[mask], target[mask]) /
                      np.dot(target[mask], target[mask]))
        rmse = float(np.sqrt(np.mean((change[mask] - target[mask])**2)))
        dose_results.extend([slope, rmse])
    joint_decoders193[ridge] = {
        "ne": coefs[:, 0], "ins": coefs[:, 1],
        "ne_r2": ne_r2, "ins_rmse": ins_rmse,
        "dose_metrics": dose_results
    }
    print("ridge", ridge, "NE R2", round(float(ne_r2), 5),
          "insulin preservation RMSE", round(ins_rmse, 5),
          "half/double slope,RMSE", np.round(dose_results, 5))

print("\nRESPONSE DIRECTION CHECK")
print("insulin outer reconstruction error",
      float(np.linalg.norm(delta_b187 -
                           np.outer(ins_direction187, old_decoder187))))
print("NE target [mean,std,min,max]",
      [float(np.mean(target_ne187)), float(np.std(target_ne187)),
       float(np.min(target_ne187)), float(np.max(target_ne187))])
for key, value in directions_ne187.items():
    if isinstance(value, dict):
        print("NE direction", key, summary189(value))
    else:
        direction = np.asarray(as_np184(value)).reshape(-1)
        print("NE direction", key, "norm", float(np.linalg.norm(direction)),
              "target RMSE",
              float(np.sqrt(np.mean((J187 @ direction - target_ne187)**2))))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 193\n"
        "Run-192 ridge .001/strength 100 improved raw held-out NE dose slopes "
        "to .989/.950 with factual signal RMSE .130. Constructed an UNSAVED "
        "amplitude-20 dual channel using this decoder and refitted joint NE "
        "decoding/insulin-preservation vectors on stay-disjoint training bundles. "
        "CPU weights restored to submitted edits in finally; no saved weights changed. "
        "Transmission validation is separate from record-based causal identification.\n"
    )
```