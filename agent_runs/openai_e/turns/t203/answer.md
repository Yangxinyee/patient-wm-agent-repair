The rebuilt static input matches `base162`, while the older reference adds the numeric projection again. I’ll decode the same original-model states with both versions and compare their means and NLLs with the GPU. The version that agrees across events will be the calibration reference.

```python
q174, zp174 = decoder_query(kind="gelu", gap_style="log", type_id=1)
print("REFERENCE DOUBLE-PROJECTION ERROR",
      float(torch.max(torch.abs(static_all122 - static168 - torch.tensor(projection173)))))
print("PRIOR LOG-SD RANGE",
      float(zp174[:, 32:].min()), float(zp174[:, 32:].max()))
print("TARGET BIN AGREEMENT", np.mean(b168 == aligned_meta168.target_bin.to_numpy()))

decoder_tests174 = {}
for label174, static174 in (
    ("one_projection", static168),
    ("two_projections", static168 + torch.tensor(projection173))
):
    mean174, hidden174, logits174 = head_calc(q174 + static174)
    prob174 = logits174.softmax(-1)
    loss174 = -prob174[torch.arange(len(b168)), torch.as_tensor(b168)].log()
    local174 = pd.DataFrame({
        "event": gpu168.event.to_numpy(),
        "mean": mean174.numpy(), "nll": loss174.numpy(),
        "mean_error": mean174.numpy() - gpu168.predicted_mean.to_numpy(),
        "nll_error": loss174.numpy() - gpu168.nll.to_numpy()
    })
    decoder_tests174[label174] = (prob174, hidden174, logits174, local174)
    print("\n", label174, {
        "NLL": float(loss174.mean()),
        "mean_RMSE": float(np.sqrt(np.mean(local174.mean_error**2))),
        "NLL_error_RMSE": float(np.sqrt(np.mean(local174.nll_error**2)))
    })
    print(local174.groupby("event").agg(
        nll=("nll", "mean"),
        mean_bias=("mean_error", "mean"),
        mean_rmse=("mean_error", lambda x: np.sqrt(np.mean(x*x))),
        nll_bias=("nll_error", "mean")
    ).to_string())

print("\nEFFECT EVIDENCE, BOUNDED")
for marker174 in ("## Run 69:", "## Run 73:", "## Run 75:", "## Run 76:", "## Run 77:"):
    pos174 = nt170.find(marker174)
    if pos174 >= 0:
        end174 = nt170.find("\n## ", pos174 + 3)
        text174 = nt170[pos174:end174 if end174 >= 0 else None]
        print(text174[:850])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 174: GPU arbitration of register assembly\n"
            "Rebuilt static equals base162; retained static_all122 contains an additional numeric projection. "
            "Comparing both original-state decoders directly with GPU factual means and NLLs. "
            "No weights changed. Python runs: 174; forwards: 22.\n")
```