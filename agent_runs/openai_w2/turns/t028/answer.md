The query type is now clear, and MAP reconstruction errors are consistent with latent sampling noise. Glucose-finger predictions still show a systematic bias. I’ll test the remaining register flags and bin-boundary convention; if those explain the bias, we can use the reconstructed head to calculate a targeted correction.

```python
import itertools
import numpy as np
import pandas as pd
import time

t0 = time.time()
register_audit = []
flag_columns = [0, 1, 2, 4, 5, 7]
for bits in itertools.product([0, 1], repeat=len(flag_columns)):
    flags = tuple(c for c, bit in zip(flag_columns, bits) if bit)
    for side in ["left", "right"]:
        cfg = dict(reconstruction_cfg, flags=flags)
        a = local_value_head(
            audit_rows, cfg, bin_side=side, zmode="prior_mc", nsamples=64
        )
        err = a["mean"] - audit_mean
        rm = float(np.sqrt(np.mean(err ** 2)))
        rn = float(np.sqrt(np.mean((a["nll"] - audit_nll) ** 2)))
        register_audit.append(dict(
            flags=flags, bin_side=side, mean_rmse=rm, nll_rmse=rn,
            mean_bias=float(err.mean()), score=rm + 3 * rn, cfg=cfg
        ))

register_table = pd.DataFrame(register_audit).sort_values("score")
print("=== REGISTER FLAG AND BIN AUDIT ===")
print(register_table.drop(columns="cfg").head(10).round(4).to_string(index=False))

activation_audit = []
for row in register_table.head(3).itertuples():
    for gap_act, prior_act, value_act in itertools.product(
        ["gelu", "silu"], ["gelu", "silu"], ["gelu", "relu", "silu"]
    ):
        cfg = dict(row.cfg, gap_activation=gap_act,
                   prior_activation=prior_act, value_activation=value_act)
        a = local_value_head(
            audit_rows, cfg, bin_side=row.bin_side,
            zmode="prior_mc", nsamples=128
        )
        rm = float(np.sqrt(np.mean((a["mean"] - audit_mean) ** 2)))
        rn = float(np.sqrt(np.mean((a["nll"] - audit_nll) ** 2)))
        activation_audit.append(dict(
            flags=row.flags, bin_side=row.bin_side, gap=gap_act,
            prior=prior_act, value=value_act, mean_rmse=rm,
            nll_rmse=rn, score=rm + 3 * rn, cfg=cfg
        ))

activation_table = pd.DataFrame(activation_audit).sort_values("score")
print("\n=== ACTIVATION AUDIT ===")
print(activation_table.drop(columns="cfg").head(8).round(4).to_string(index=False))

reconstruction_cfg = dict(activation_table.iloc[0]["cfg"])
reconstruction_side = activation_table.iloc[0]["bin_side"]
reconstruction_audit = local_value_head(
    audit_rows, reconstruction_cfg, bin_side=reconstruction_side,
    zmode="prior_mc", nsamples=512, return_features=True
)
err_table = audit_rows[["event"]].copy()
err_table["error"] = reconstruction_audit["mean"] - audit_mean
err_table["error_sq"] = err_table.error ** 2
err_table["expected_error_sq"] = (
    reconstruction_audit["sampling_mean_sd"] ** 2 * (1 / 4 + 1 / 512)
)
print("\n=== BEST CONFIGURATION ===")
print(reconstruction_cfg, "bin_side:", reconstruction_side)
print(err_table.groupby("event").agg(
    n=("error", "size"), bias=("error", "mean"),
    mse=("error_sq", "mean"),
    expected_sampling_mse=("expected_error_sq", "mean")
).round(4).to_string())
print("seconds:", round(time.time() - t0, 1))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 24: register and activation audit\n"
        "MAP discrepancies can be explained by four-draw latent sampling. "
        "Searching glucose register flags and boundary conventions. Best config: "
        + repr(reconstruction_cfg) + "; bin_side=" + reconstruction_side + "\n"
        + err_table.groupby("event").agg(
            bias=("error", "mean"), mse=("error_sq", "mean"),
            expected_sampling_mse=("expected_error_sq", "mean")
        ).to_string() + "\n"
    )
```