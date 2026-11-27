The decoder maps each prelude slot directly to the insulin innovation. I’ll connect it using the smaller glucose-selective response direction. This is a provisional rank-one edit: factual NLL must stay within budget, and rollouts must move toward the record-derived effects rather than become unstable.

```python
connection_scale123 = 1.0
connection_ridge123 = 0.01
connection_decoder123 = np.asarray(chosen_channel116["decoder"], dtype=np.float64)
connection_direction123 = np.asarray(directions118[connection_ridge123], dtype=np.float64)

connection_delta123 = (
    connection_scale123
    * np.outer(connection_direction123, connection_decoder123)
).astype(np.float32)
connection_weight123 = (
    np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
    + connection_delta123
)

save_edit("core.b_in.weight", connection_weight123)

# Keep the verified CPU reconstruction synchronized with the submitted weights.
rW = {name: torch.as_tensor(load(name), dtype=torch.float32) for name in NAMES}
rp_linear = original_linear118

decoded_slots123 = (
    np.asarray(chosen_channel116["capture"]["ep"], dtype=np.float64)
    @ connection_decoder123
)
print("Connection scale:", connection_scale123)
print("Rank-one update norm:", np.linalg.norm(connection_delta123))
print("Decoded factual innovation quantiles:",
      np.quantile(decoded_slots123.mean(axis=1), [0, .01, .5, .99, 1]))
print("Across-slot decoder SD:", decoded_slots123.std(axis=1).mean())
print("Current edits:", edited())

with open("/work/notes.md", "a") as f123:
    f123.write(
        "\n## Run 123: provisional insulin response connection\n"
        "Recovered exact interfaces: rW is the CPU tensor dictionary; transition73 "
        "is the verified one-bundle transition; transition_with_b110 injects an "
        "additive core.b_in output; probe_means95 evaluates the next reading. "
        "chosen_channel116['decoder'] has shape (192,), decoding prelude E per slot. "
        "directions118[0.01] is the smaller glucose-selective response direction "
        "(norm 23.695), calibrated to immediate record estimate -2.04924/rate. "
        "Saved core.b_in.weight = original + outer(direction, decoder), scale 1.0, "
        "in addition to the nine channel edits. This remains provisional pending "
        "independent GPU factual and rollout tests; sustained kernels are not yet "
        "treated as established ground truth.\n"
    )
```
