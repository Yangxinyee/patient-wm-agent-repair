The local helpers use a cached weight dictionary, so they may still be evaluating original weights. I’ll rebuild the reading features explicitly with the current edit and inspect what the value head actually receives. A reproducible dose signal there would support a targeted correction; otherwise the transport needs repair first.

```python
print("=== RELEVANT FILES ===")
for folder133 in ["/work/fwd_11", "/work/fwd_12", "/work/fwd_13"]:
    p133 = Path(folder133)
    print(folder133, [p.name for p in p133.iterdir()] if p133.exists() else "missing")

print("=== HELPER RETURNS AND CONSTANTS ===")
for name133 in ["reading_cache", "head_features_layout", "head_mc_layout"]:
    fun133 = globals()[name133]
    print(name133, "constants:",
          [c for c in fun133.__code__.co_consts
           if isinstance(c, (str, int, float, tuple))])
print("layout:", globals().get("head_layout_spec"))
print("dataframes:",
      [(k, tuple(v.shape)) for k, v in list(globals().items())
       if isinstance(v, pd.DataFrame) and
       (k in ["train", "tune", "panel", "panel_tune", "X"] or "cache" in k.lower())])

statesTransport133 = dict(np.load("/work/fwd_13/states_edited.npz"))
print("state fields:", {k: v.shape for k, v in statesTransport133.items()})
recordsTune133 = pd.read_csv("/data/tuning.csv")
Woriginal133 = W0
Wtransport133 = {n: load(n).clone() for n in NAMES}
try:
    W0 = Wtransport133
    cacheTransport133 = reading_cache(recordsTune133, statesTransport133)
    featuresTransport133 = head_features_layout(
        cacheTransport133, head_layout_spec, subset=np.arange(12),
        nsamples=4, seed=913
    )
finally:
    W0 = Woriginal133

print("cache fields:",
      {k: tuple(v.shape) for k, v in cacheTransport133.items()
       if hasattr(v, "shape")})
print("feature fields:",
      {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
       for k, v in featuresTransport133.items()})
print("reading example:")
print(cacheTransport133["df"].head(5).to_string(index=False))

with open("/work/notes.md", "a") as f133:
    f133.write(
        "\n## Run 133: restart audit\n"
        "- Re-read architecture and earlier record-derived effects: insulin approximately "
        "-1.84 glucose units per unit/hour; NE approximately +3.2 to +3.6 MAP units "
        "per rate unit. Dynamic targets remain assumption-dependent.\n"
        "- Existing transport passes tuning-A factual NLL (+.00209), but saved "
        "private-channel slopes are very small. No causal value-head coupling yet.\n"
        "- Rebuilding local reading features with edited rather than cached original "
        "weights before further design or validation.\n"
    )
```
