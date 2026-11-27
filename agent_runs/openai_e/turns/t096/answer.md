The selected detector has low factual leakage, and the new direction reduces glucose cross-response across the calibration contexts. I’ll add a reversible norepinephrine correction scaled to the record-based one-hour effect of about +3 MAP units per dose unit. It remains provisional: the full model must pass the factual NLL limit and improve both norepinephrine rollouts.

```python
baseline_b83 = load("core.b_in.weight").clone()
baseline_delta83 = array80(
    baseline_b83 - load_original("core.b_in.weight")
).copy()

ne_beta83 = 3.0
ne_strength83 = ne_beta83 / ne_slope82
ne_delta83 = ne_strength83 * np.outer(ne_direction82, ne_feature82)
candidate_b83 = baseline_b83 + torch.from_numpy(ne_delta83.astype(np.float32))
save_edit("core.b_in.weight", candidate_b83)

np.savez("/work/ne_trial83.npz",
         baseline_delta=baseline_delta83,
         ne_delta=ne_delta83.astype(np.float32),
         feature=ne_feature82,
         direction=ne_direction82,
         strength=ne_strength83,
         beta=ne_beta83)

print("NE strength:", ne_strength83)
print("NE delta norm:", np.linalg.norm(ne_delta83))
print("Combined delta norm:", np.linalg.norm(baseline_delta83 + ne_delta83))
print("NE admission feature max:", np.max(np.abs(initial_matrix56 @ ne_feature82)))
print("Edited tensors:", edited())
print("NEXT FORWARD:")
print(json.dumps({
    "type": "factual", "model": "both", "split": "tuning",
    "stays": validation_ids80
}))

with open("/work/notes.md", "a") as file83:
    file83.write(
        "\n## Run 83: reversible insulin+NE trial installed\n"
        "Added a rank-one NE correction to the preserved insulin baseline. "
        f"Detector={chosen_key82}; multi-context MAP direction ridge=.01; "
        f"strength={ne_strength83:.6g}, NE delta norm={np.linalg.norm(ne_delta83):.6g}. "
        "Scaling uses record-based NE beta=3.0, not external medical knowledge. "
        "Trial is unaccepted pending paired factual and intervention validation. "
        "Saved /work/ne_trial83.npz. Python runs: 83; forwards: 10.\n"
    )
```