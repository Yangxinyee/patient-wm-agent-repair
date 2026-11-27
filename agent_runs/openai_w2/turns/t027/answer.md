The sampled states are aligned with earlier bundles, and the dose distributions provide usable variation. One important correction: the forward tool averages only four latent draws, so some reconstruction discrepancy is expected. I’ll test query types and register layouts, checking whether the remaining errors are unbiased and consistent with that sampling noise.

```python
import numpy as np
import pandas as pd
import time

t0 = time.time()
audit_rows = head_data.iloc[test_rows].copy()
audit_mean = audit_rows.predicted_mean.to_numpy()
audit_nll = audit_rows.nll.to_numpy()

layouts = [
    ("document_layout", dict(zcol=2, acol=1, flags=(0, 3),
                            age_mode="seconds_div10")),
    ("best_register_search", dict(zcol=6, acol=3, flags=(0,),
                                 age_mode="seconds_div10")),
]
query_audit = []
for layout_name, layout in layouts:
    for type_id in range(15):
        for status_id in [0, 1]:
            cfg = dict(head_cfg)
            cfg.update(layout)
            cfg.update(type_id=type_id, status_id=status_id,
                       value_activation="relu")
            a = local_value_head(audit_rows, cfg, zmode="prior_mean")
            rm = float(np.sqrt(np.mean((a["mean"] - audit_mean) ** 2)))
            rn = float(np.sqrt(np.mean((a["nll"] - audit_nll) ** 2)))
            query_audit.append(dict(
                layout=layout_name, type_id=type_id, status_id=status_id,
                mean_rmse=rm, nll_rmse=rn, score=rm + 3 * rn, cfg=cfg
            ))

query_table = pd.DataFrame(query_audit).sort_values("score")
print("=== DETERMINISTIC QUERY SEARCH ===")
print(query_table.drop(columns="cfg").head(10).to_string(index=False))

mc_audit = []
for row in query_table.head(6).itertuples():
    for value_activation in ["relu", "gelu", "silu"]:
        cfg = dict(row.cfg, value_activation=value_activation)
        a = local_value_head(audit_rows, cfg, zmode="prior_mc", nsamples=128)
        err = a["mean"] - audit_mean
        expected_rmse = float(np.sqrt(np.mean(
            a["sampling_mean_sd"] ** 2 * (1 / 4 + 1 / 128)
        )))
        rm = float(np.sqrt(np.mean(err ** 2)))
        rn = float(np.sqrt(np.mean((a["nll"] - audit_nll) ** 2)))
        mc_audit.append(dict(
            layout=row.layout, type_id=row.type_id, status_id=row.status_id,
            value_activation=value_activation, mean_bias=float(err.mean()),
            mean_rmse=rm, expected_sampling_rmse=expected_rmse,
            nll_rmse=rn, score=rm + 3 * rn, cfg=cfg
        ))

mc_table = pd.DataFrame(mc_audit).sort_values("score")
print("\n=== MONTE CARLO AUDIT ===")
print(mc_table.drop(columns="cfg").head(8).round(4).to_string(index=False))

reconstruction_cfg = dict(mc_table.iloc[0]["cfg"])
reconstruction_audit = local_value_head(
    audit_rows, reconstruction_cfg, zmode="prior_mc",
    nsamples=256, return_features=True
)
err_table = audit_rows[["event"]].copy()
err_table["error"] = reconstruction_audit["mean"] - audit_mean
err_table["error_sq"] = err_table.error ** 2
err_table["expected_error_sq"] = (
    reconstruction_audit["sampling_mean_sd"] ** 2 * (1 / 4 + 1 / 256)
)
print("\n=== BEST CONFIGURATION ===")
print(reconstruction_cfg)
print("\n=== ERRORS BY CHANNEL ===")
print(err_table.groupby("event").agg(
    n=("error", "size"), bias=("error", "mean"),
    mse=("error_sq", "mean"),
    expected_sampling_mse=("expected_error_sq", "mean")
).round(4).to_string())
print("seconds:", round(time.time() - t0, 1))
```