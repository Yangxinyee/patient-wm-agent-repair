Scale 0.5 still predicts only −2.25 for doubled insulin and +2.75 for halved insulin, while norepinephrine remains nearly unchanged. Scaling alone is therefore inadequate. I’ll separate two possible causes: dose information being compressed before it reaches the state, and an incorrect response over time. Accurate dose decoding but weak outcomes would point to the second; compressed decoding would require repairing the input channel first.

```python
print("=== EARLIER DESIGN NOTES ===")
notes142 = Path("/work/notes.md").read_text()
print(notes142[:4800])

print("\n=== ACTION / CHANNEL INTERFACES ===")
for name142 in [
    "controlled_action_repair", "action_atom_vectors_repair",
    "action_fixture_repair", "action_ctx_repair",
    "gated_innovation94", "capture_transition102",
    "capture_ep112", "transition_with_z95", "head_features_fixed",
    "head_predictions_repair", "probe_indices95", "drug_indices95",
    "dose_probe104", "channel_fits112", "reader_direction112",
    "channel_direction112", "amplitude112", "q112", "q118",
    "actual_insulin95"
]:
    value142 = globals().get(name142)
    if callable(value142):
        print(name142, inspect.signature(value142),
              "dependencies:", value142.__code__.co_names)
    elif isinstance(value142, dict):
        print(name142, "keys:", list(value142)[:20])
    elif isinstance(value142, (np.ndarray, torch.Tensor)):
        arr142 = torch.as_tensor(value142).float()
        print(name142, "shape", tuple(arr142.shape),
              "mean/std", float(arr142.mean()), float(arr142.std()))
    else:
        print(name142, value142 if np.isscalar(value142) else type(value142).__name__)

print("\n=== TRAINING FIXTURE ===")
for name142 in ["E65", "U65", "reference_x125", "fact_state_repair",
                "fact_prev_repair", "fact_dt_repair"]:
    print(name142, getattr(globals().get(name142), "shape", None))
print("function names:",
      sorted(n142 for n142, v142 in list(globals().items())
             if callable(v142) and hasattr(v142, "__code__")
             and any(s142 in n142.lower() for s142 in
                     ["action", "capture", "innovation", "posterior", "replay", "nll"])))

with open("/work/notes.md", "a") as f142:
    f142.write(
        "\n## Run 142\n"
        "GPU scale-0.5 factual first-100 tuning dNLL=+0.00037. "
        "Rollouts: insulin x2=-2.252, half=+2.748; NE x2=+0.090, half=-0.290. "
        "Scaling alone remains inadequate. Investigating input dose compression "
        "versus response dynamics before another edit. Scale 0.5 remains saved.\n"
    )
```