The record kernels imply substantially larger effects than the original rollouts, but the CPU head still differs by about four units on average. That is too large to trust yet. I’ll test the query-type and status conventions next. A sharp improvement would identify an input-mapping error; otherwise I’ll investigate latent sampling and other missing details.

```python
print("=== OFFICIAL ORIGINAL ROLLOUT SUMMARY ===")
print(Path("/work/fwd_1/summary.json").read_text()[:3500])
print("\nOriginal means with alternative weighting:")
for e24, r24 in roll22.groupby("edit"):
    reached24 = r24.loc[r24.reached]
    print(e24, "rows", len(r24), "reached", len(reached24),
          "all-row mean", r24["shift"].mean(),
          "reached-row mean", reached24["shift"].mean(),
          "reached stay-mean", reached24.groupby("stay_id")["shift"].mean().mean())

print("\nStatus embedding norms:", w22("last_status.weight").norm(dim=1).tolist())
print("Prior output and last_numeric column norms:",
      w22("z_prior.2.weight").norm(dim=1).reshape(2, 32).mean(1).tolist(),
      w22("last_numeric.weight").norm(dim=0).tolist())

base_no_status24 = base22 - w22("last_status.weight")[0]
head_tests24 = []
start24 = time.time()
for prior_act24 in ["gelu", "silu", "relu"]:
    gap_input24 = torch.tensor(np.log(dt22) / 10., dtype=torch.float32)[:, None]
    ctx24 = out22 + flow_summary22 + mlp22(gap_input24, "gap", prior_act24)
    prior_raw24 = mlp22(ctx24, "z_prior", prior_act24)
    view24 = ctx24 + lin22(prior_raw24[:, :32], "z_head")
    for age_unit24 in ["seconds", "hours"]:
        age24 = np.log1p(age_hours22 * (3600 if age_unit24 == "seconds" else 1)) / 10.
        nf24 = np.zeros((len(ds22), 8), np.float32)
        nf24[:, 0], nf24[:, 1] = has22, age24
        nf24[:, 2], nf24[:, 3] = last_z22, has22
        register24 = lin22(torch.tensor(nf24), "last_numeric")
        for type24 in range(15):
            q24 = norm22(view24 + w22("type_emb.weight")[type24], "query_norm")
            for status24 in [-1, 0, 1]:
                status_vector24 = 0. if status24 == -1 else w22("last_status.weight")[status24]
                wc24 = q24 + base_no_status24 + register24 + status_vector24
                for head_act24 in ["gelu", "silu", "relu"]:
                    logits24 = mlp22(wc24, "attr_heads.value_bin", head_act24) + transition22
                    probs24 = logits24.softmax(-1)
                    mean24 = (probs24[:, 1:] * centres22).sum(-1)
                    nll24 = -probs24[torch.arange(len(ds22)), observed_bin22].log()
                    rmse24 = np.sqrt(np.mean((mean24.numpy() - reference_mean22)**2))
                    mae24 = np.mean(np.abs(nll24.numpy() - reference_nll22))
                    head_tests24.append((rmse24, mae24, prior_act24, age_unit24,
                                         head_act24, type24, status24))
head_tests24.sort()
print("\n=== BEST LOOKUP TESTS ===")
for result24 in head_tests24[:12]:
    print(result24)
head_spec24 = head_tests24[0]
_, _, pa24, au24, ha24, ti24, st24 = head_spec24
ctx24 = out22 + flow_summary22 + mlp22(
    torch.tensor(np.log(dt22) / 10., dtype=torch.float32)[:, None], "gap", pa24)
prior_raw24 = mlp22(ctx24, "z_prior", pa24)
age24 = np.log1p(age_hours22 * (3600 if au24 == "seconds" else 1)) / 10.
nf24 = np.zeros((len(ds22), 8), np.float32)
nf24[:, 0], nf24[:, 1], nf24[:, 2], nf24[:, 3] = has22, age24, last_z22, has22
q24 = norm22(ctx24 + lin22(prior_raw24[:, :32], "z_head")
             + w22("type_emb.weight")[ti24], "query_norm")
wc24 = (q24 + base_no_status24 + lin22(torch.tensor(nf24), "last_numeric")
        + (0. if st24 == -1 else w22("last_status.weight")[st24]))
hidden24 = act22(lin22(wc24, "attr_heads.value_bin.0"), ha24)
logits24 = lin22(hidden24, "attr_heads.value_bin.2") + transition22
prob24 = logits24.softmax(-1)
prediction24 = (prob24[:, 1:] * centres22).sum(-1).numpy()
nll_prediction24 = -prob24[torch.arange(len(ds22)), observed_bin22].log().numpy()

print("\n=== BEST RESIDUALS BY SENSOR ===")
for event24 in ds22.event.unique():
    mask24 = ds22.event.eq(event24).to_numpy()
    error24 = prediction24[mask24] - reference_mean22[mask24]
    print(event24, "n", mask24.sum(), "bias", round(error24.mean(), 4),
          "RMSE", round(np.sqrt(np.mean(error24**2)), 4),
          "NLL MAE", round(np.mean(np.abs(nll_prediction24[mask24]
                                         - reference_nll22[mask24])), 4))
print("Prior mean SD:", prior_raw24[:, :32].std().item(),
      "log-SD quantiles:", np.quantile(prior_raw24[:, 32:].numpy(), [.01, .5, .99]))
print("No-value probability quantiles:", np.quantile(prob24[:, 0].numpy(), [.5, .99]))
print("seconds", round(time.time() - start24, 2))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 24\n"
            "Testing all query-type indices and optional last-status embeddings. "
            "Best head lookup specification: " + repr(head_spec24) + ". "
            "Record targets differ under row versus stay weighting; official forward summary inspected. "
            "No edits; CPU reconstruction must be validated before deriving head edits.\n")
```
