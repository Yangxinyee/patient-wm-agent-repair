The rebuild stopped because I passed the layout dictionary where the helper expects its tuple. No weights changed. I’ll correct that, compare the local head predictions with the GPU results, and align each reading with its preceding exposure. This will test whether the features are trustworthy enough for dose decoding.

```python
layout134 = head_layout_spec["layout"]
ageunit134 = head_layout_spec["ageunit"]
factualE134 = pd.read_csv("/work/fwd_12/factual_edited.csv")
factualO134 = pd.read_csv("/work/fwd_12/factual_original.csv")
statesOriginal134 = dict(np.load("/work/fwd_13/states_original.npz"))

try:
    W0 = Wtransport133
    cacheTransport133 = reading_cache(
        recordsTune133, statesTransport133, factualE134
    )
    featuresTransport134 = head_features_layout(
        cacheTransport133, layout134, ageunit=ageunit134,
        nsamples=4, seed=913
    )
    W0 = Woriginal133
    cacheOriginal134 = reading_cache(
        recordsTune133, statesOriginal134, factualO134
    )
    featuresOriginal134 = head_features_layout(
        cacheOriginal134, layout134, ageunit=ageunit134,
        nsamples=4, seed=913
    )
finally:
    W0 = Woriginal133

print("=== FEATURE SHAPES ===")
print({k: tuple(v.shape) for k, v in featuresTransport134.items()
       if hasattr(v, "shape")})
print("=== LOCAL / GPU CHECK ===")
for label134, cc134, ff134 in [
    ("original", cacheOriginal134, featuresOriginal134),
    ("transport", cacheTransport133, featuresTransport134)
]:
    mm134 = as_np(ff134["means"]).reshape(-1)
    nn134 = as_np(ff134["nll"]).reshape(-1)
    print(label134, {
        "local_nll": float(nn134.mean()),
        "gpu_nll": float(cc134["gpu_nll"].mean()),
        "mean_rmse_to_gpu": float(np.mean(
            (mm134 - cc134["gpu_mean"])**2)**0.5)
    })

df134 = cacheTransport133["df"]
exposureIndex134 = pd.MultiIndex.from_arrays(
    [df134.stay_id.to_numpy(),
     np.rint(df134.hour.to_numpy()).astype(int) - 1],
    names=panel_tune.index.names
)
doses134 = panel_tune.reindex(exposureIndex134)[
    ["insulin", "norepinephrine"]
].fillna(0).to_numpy()
print("=== ALIGNED EXPOSURES ===")
print("reading count:", len(df134),
      "positive exposures:", (doses134 > 0).sum(axis=0))
print(df134.head(5).to_string(index=False))
print("base private mean/std:",
      cacheTransport133["base"][:, R125].mean(axis=0),
      cacheTransport133["base"][:, R125].std(axis=0))

with open("/work/notes.md", "a") as f134:
    f134.write(
        "\n## Runs 130–134: restart and head audit\n"
        "- Re-read architecture and earlier record-derived effects. Dynamic "
        "targets remain conditional on identification and response-form assumptions.\n"
        "- Run 133 stopped on a layout-dictionary/tuple mismatch; no weights changed.\n"
        "- Run 134 rebuilt reading features using explicitly edited weights and "
        "aligned readings with preceding-hour exposures. No causal head edit yet.\n"
    )
```
