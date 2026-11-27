The head count is correct, and the replay’s bundle encodings and posterior means match the exports exactly. That rules out those suspected causes. I’ll invert the coda and output projection to isolate the recurrent update; a mismatch concentrated there would support an update-equation error, while unstable inversion would make this test inconclusive.

```python
import torch, numpy as np
torch.set_num_threads(8)

# Invert the near-identity coda by fixed-point iteration.
rx_y0131 = rx_targeth.clone()
for _ in range(40):
    rx_coda131 = rx_layer2(rx_y0131, "core.coda.0", "gelu")
    rx_y0131 = rx_targeth - (rx_coda131 - rx_y0131)
rx_coda_err131 = float(
    (rx_layer2(rx_y0131, "core.coda.0", "gelu") - rx_targeth)
    .square().mean().sqrt()
)
print("Coda inversion RMSE:", rx_coda_err131)

rx_C131 = rx_W["core.c_out.weight"].double()
rx_sv131 = torch.linalg.svdvals(rx_C131)
print("c_out singular-value range:", float(rx_sv131.min()), float(rx_sv131.max()))
rx_pre131 = torch.linalg.solve(
    rx_C131, rx_y0131.reshape(-1, 192).double().T
).T.reshape_as(rx_targeth).float()

rx_E131 = rx_norm(
    rx_layer2(rx_tokens126, "core.prelude.0", "gelu")[:, :4],
    "core.prelude_norm"
)
rx_base131 = rx_hflow + rx_E131
rx_U131 = rx_base131.clone()
for layer in ("core.blocks.0.0", "core.blocks.0.1"):
    rx_U131 = rx_layer2(
        rx_U131, layer, "silu", rx_mem, rx_ages, rx_memvalid,
        "before_kv", "gelu", 1.0
    )
rx_resid131 = rx_U131 - rx_base131
rx_drive131 = rx_delta126[None, None, :] * rx_linear(rx_E131, "core.b_in")
rx_expected131 = (
    rx_abar126[None, None, :] * rx_hflow + rx_drive131 + rx_resid131
)

print("Pre-coda reconstructed target std:", float(rx_pre131.std()))
print("Expected recurrent-update std:", float(rx_expected131.std()))
print("Pre-coda update RMSE:", float(
    (rx_pre131-rx_expected131).square().mean().sqrt()
))

rx_hprev131 = rx_prev[:, 192:960].reshape(-1, 4, 192)
rx_features131 = torch.stack([
    (rx_abar126[None, None, :] * rx_hflow).reshape(-1),
    rx_drive131.reshape(-1),
    rx_resid131.reshape(-1),
    rx_E131.reshape(-1),
    rx_hprev131.reshape(-1),
    torch.ones(rx_pre131.numel()),
], dim=1).double()
rx_fit131 = torch.linalg.lstsq(
    rx_features131, rx_pre131.reshape(-1).double()
).solution
print("Coefficients [abar*hflow, drive, block residual, E, hprev, intercept]:",
      rx_fit131.numpy())
print("Fitted structural RMSE:", float(
    (rx_features131 @ rx_fit131-rx_pre131.reshape(-1).double())
    .square().mean().sqrt()
))

rx_a_est131 = (
    (rx_hflow * (rx_pre131-rx_drive131-rx_resid131)).sum(dim=(0, 1))
    / rx_hflow.square().sum(dim=(0, 1)).clamp_min(1e-8)
)
print("Estimated a_bar quantiles:",
      np.quantile(rx_a_est131.numpy(), [0, .1, .5, .9, 1]))
print("Estimated versus documented a_bar RMSE:",
      float((rx_a_est131-rx_abar126).square().mean().sqrt()))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n\n## Run 131\n"
        "Replay encodings/posteriors are exact and attention head count is correct. "
        f"Coda inversion RMSE={rx_coda_err131:.6g}; "
        "isolated recurrent update via inverse output projection and fitted "
        "structural coefficients. No weights edited. "
        "Counts: 131 Python runs, 6 forward passes.\n"
    )
```