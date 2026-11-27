The clean drug-start reconstruction is close, but its error is larger than the exact reading and END matches. I’ll isolate whether that comes from the older action helper or time-gap rounding. Agreement with the saved latent mean will confirm the encoding; a persistent mismatch would mean it is not yet safe to use for editing.

```python
print("=== ENCODER CONSTANTS ===")
for helper_name in ["single_set57", "latent_action_encode",
                    "latent_encode_batch", "repair_mlp_act", "complete_atom57"]:
    helper = globals()[helper_name]
    print(helper_name, helper.__code__.co_consts)

print("\n=== SET-ENCODER TENSOR NAMES ===")
for name in NAMES:
    if name.startswith("encode.") or name.startswith("encode_action."):
        print(name, SHAPES[name])

i64, starts64, ends64 = action_cases50[0]
row64 = starts64[0]
atom64 = complete_atom57(
    row64, kind=0, ekind=2,
    unit={"insulin": 3, "norepinephrine": 4}[row64["event"]],
    rate_unit={"insulin": 2, "norepinephrine": 3}[row64["event"]],
    numbers=True
)
u64 = single_set57(atom64.reshape(1, 192), "encode_action")
e64 = empty_set58("encode").reshape(1, 192)
target64 = torch.as_tensor(latent_state[i64, -32:], dtype=torch.float32).reshape(1, 32)
previous64 = torch.as_tensor(latent_state[i64 - 1], dtype=torch.float32)
raw_dt64 = float((latent_hour[i64] - latent_hour[i64 - 1]) * 3600)

print("\n=== CLEAN-START ENCODING / GAP CHECK ===")
print("old-versus-complete action RMS:",
      float(torch.sqrt(torch.mean((u64 - u63) ** 2))))
checks64 = []
for dt_name64, dt64 in [
    ("raw_seconds", raw_dt64),
    ("rounded_seconds", float(round(raw_dt64))),
    ("rounded_minutes", float(round(raw_dt64 / 60) * 60))
]:
    h64 = previous64[192:960].reshape(1, 4, 192)
    flowed64 = h64 * torch.exp(
        -dt64 * torch.exp(repair_weights["core.flow.a"])
    )
    for gap_name64, gap64 in [
        ("log_dt_div10", np.log(max(dt64, 1e-6)) / 10),
        ("log1p_dt_div10", np.log1p(max(dt64, 0)) / 10)
    ]:
        gap_vec64 = repair_linear(
            F.gelu(repair_linear(torch.tensor([[gap64]], dtype=torch.float32), "gap.0")),
            "gap.2"
        )
        context64 = previous64[:192].reshape(1, 192) + \
            repair_norm(flowed64.mean(1), "core.out_norm") + gap_vec64
        pred64 = post_mean59(context64, e64, u64)
        rms64 = float(torch.sqrt(torch.mean((pred64 - target64) ** 2)))
        checks64.append((rms64, dt_name64, gap_name64))
print(sorted(checks64))
print("complete atom, retained context RMS:",
      float(torch.sqrt(torch.mean(
          (post_mean59(latent_context([i64]), e64, u64) - target64) ** 2
      ))))

print("\n=== OBSERVATION SAMPLE STRUCTURE ===")
print("obs_samples52:", small_description62(obs_samples52[:1])
      if isinstance(obs_samples52, (list, tuple)) else small_description62(obs_samples52))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 64; forward 6 — clean-start precision check\n"
        "- Old action-helper reconstruction has RMS 3.79e-4 despite exact reading/END "
        "matches. Comparing explicit categorical encoding and gap rounding before edits.\n"
        "- No weights changed.\n"
    )
```