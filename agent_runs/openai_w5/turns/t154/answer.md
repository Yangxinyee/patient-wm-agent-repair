The local paired loss roughly matches the GPU penalty, though individual means differ with latent draws. I’ll now make the dose channel distinct from additive head inputs and amplify it, while preserving atom-encoder preactivations by linear compensation. This remains a capacity test: excessive factual loss would reject it before any causal correction is added.

```python
baseline135 = {n: load(n).clone() for n in NAMES}
baselineEdited135 = set(edited())
candidateClean135 = {n: v.clone() for n, v in baseline135.items()}
changed135 = set()

def equal_columns135(name):
    t = candidateClean135[name]
    for a, b in [R125[:2], R125[2:]]:
        m = (t[:, int(a)] + t[:, int(b)]) / 2
        t[:, int(a)] = m
        t[:, int(b)] = m
    changed135.add(name)

def equal_rows135(name):
    t = candidateClean135[name]
    for a, b in [R125[:2], R125[2:]]:
        m = (t[int(a)] + t[int(b)]) / 2
        t[int(a)] = m
        t[int(b)] = m
    changed135.add(name)

for name135 in [
    "core.out_norm.weight", "core.out_norm.bias",
    "query_norm.weight", "query_norm.bias",
    "gap.2.weight", "gap.2.bias",
    "z_head.weight", "z_head.bias",
    "last_numeric.weight", "last_numeric.bias"
]:
    assert name135 in candidateClean135, name135
    equal_rows135(name135)

for name135 in [
    "type_emb.weight", "embed.code.weight",
    "embed.bins.value.weight", "embed.bins.rate.weight"
]:
    equal_columns135(name135)

for name135 in NAMES:
    if name135.startswith("last_status") and (
        candidateClean135[name135].ndim == 2
        and candidateClean135[name135].shape[1] == 192
    ):
        equal_columns135(name135)

gain135 = 400.0
candidateClean135["core.b_in.weight"][R125] *= gain135
changed135.add("core.b_in.weight")

# Preserve first-layer arithmetic for every primitive atom embedding component.
primitiveNames135 = []
Boriginal135, Bclean135 = [], []
for name135 in NAMES:
    if not name135.startswith("embed."):
        continue
    old135 = baseline135[name135]
    new135 = candidateClean135[name135]
    if old135.ndim == 2 and old135.shape[1] == 192:
        Boriginal135.append(as_np(old135).astype(np.float64))
        Bclean135.append(as_np(new135).astype(np.float64))
        primitiveNames135.append(name135)
    elif name135 == "embed.numeric.weight":
        Boriginal135.append(as_np(old135).T.astype(np.float64))
        Bclean135.append(as_np(new135).T.astype(np.float64))
        primitiveNames135.append(name135)
    elif name135 == "embed.numeric.bias":
        Boriginal135.append(as_np(old135)[None].astype(np.float64))
        Bclean135.append(as_np(new135)[None].astype(np.float64))
        primitiveNames135.append(name135)
Boriginal135 = np.concatenate(Boriginal135)
Bclean135 = np.concatenate(Bclean135)
print("primitive matrix shape/rank:",
      Bclean135.shape, np.linalg.matrix_rank(Bclean135))

for name135 in ["encode.mlp1.0.weight", "encode_action.mlp1.0.weight"]:
    old135 = as_np(baseline135[name135]).astype(np.float64)
    target135 = Boriginal135 @ old135.T
    correction135 = np.linalg.lstsq(
        Bclean135, target135 - Bclean135 @ old135.T, rcond=1e-9
    )[0].T
    new135 = torch.tensor(old135 + correction135, dtype=torch.float32)
    error135 = Bclean135 @ as_np(new135).astype(np.float64).T - target135
    print(name135, {
        "preactivation_rmse": float(np.mean(error135**2)**0.5),
        "preactivation_max_error": float(np.max(np.abs(error135))),
        "relative_weight_change": float(np.linalg.norm(correction135)
                                       / np.linalg.norm(old135))
    })
    assert np.max(np.abs(error135)) < 1e-3
    candidateClean135[name135] = new135
    changed135.add(name135)

# Remove incidental use of the new pair differences by the old value head.
equal_columns135("attr_heads.value_bin.0.weight")

for name135 in sorted(changed135):
    save_edit(name135, candidateClean135[name135])

print("Installed capacity-test tensors:", sorted(changed135))
print("Transport gain:", gain135, "Total edited tensors:", len(edited()))
print("No causal logit correction installed.")

with open("/work/notes.md", "a") as f135:
    f135.write(
        "\n## Run 135: clean-pair capacity candidate\n"
        "- Equalized paired coordinates in out/query normalization and additive "
        "head inputs; removed incidental old value-head pair sensitivity.\n"
        "- Amplified private dose transport by 400. This is not a causal calibration.\n"
        "- Compensated both atom encoders by linear algebra to preserve primitive "
        "embedding preactivations. No gradients or optimizer used.\n"
        "- Candidate requires paired GPU factual validation; retain only if the "
        "+.005 value-NLL constraint holds with room for a causal correction.\n"
    )
```
