The repaired residual estimate is too uncertain to establish a missing factual effect. The corrected detectors also retain substantial factual leakage, so scaling them up risks another NLL failure. I’ll test the posterior network’s hidden features next: they see patient context and drug input directly. Lower leakage with balanced doubling/halving sensitivity would support this alternative.

```python
import torch
import torch.nn.functional as F
import dis

with open("/work/notes.md", "a") as f101:
    f101.write("\n## Run 100 results and Run 101: posterior-feature alternative\n"
               f"Corrected original residual audit: n={len(ii100)}, beta={beta100:.4f}, "
               f"cluster SE={se100:.4f}; inconclusive diagnostic. Corrected prelude-hidden "
               "and b_in detectors have factual leakage around .37-.69 raw-dose units "
               "per unit doubling sensitivity; not yet safe to scale. Run100 completed "
               "fits but failed during its final inventory. Inspecting posterior features "
               "as a different closed-form edit route. No weights changed. "
               "Python runs: 101; forwards: 12.\n")

def asnp101(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("LOCAL CONTEXT ARRAYS")
for n101, o101 in list(globals().items()):
    if isinstance(o101, (np.ndarray, torch.Tensor)):
        sh101 = tuple(o101.shape)
        if sh101 in ((32,), (31,), (32, 192), (32, 4, 192), (4356, 192), (4356, 4, 192)):
            print(n101, sh101)
    elif "config" in n101.lower() and isinstance(o101, (tuple, list, dict)):
        print(n101, str(o101)[:200])

print("\nUPDATE RETURN CONTRACT")
print([(i101.opname, i101.argrepr) for i101 in list(dis.get_instructions(core_update_local))[-12:]])
print("src45", np.asarray(src45))
print("posterior weight shapes", tuple(W0["z_post.0.weight"].shape),
      tuple(W0["z_post.2.weight"].shape))
print("cf_frame columns", list(cf_frame.columns))
print("W0 b_in distances: current, original",
      float(np.linalg.norm(asnp101(W0["core.b_in.weight"]) - asnp101(load("core.b_in.weight")))),
      float(np.linalg.norm(asnp101(W0["core.b_in.weight"]) - asnp101(load_original("core.b_in.weight")))))

# Find the retained all-bundle context by matching the known drug-bundle context.
candidates101 = []
for n101, o101 in list(globals().items()):
    if isinstance(o101, (np.ndarray, torch.Tensor)) and tuple(o101.shape) == (4356, 192):
        error101 = float(np.sqrt(np.mean((asnp101(o101)[idx93] - ctx96) ** 2)))
        candidates101.append((error101, n101))
print("\nCONTEXT MATCHES", sorted(candidates101)[:8])

if candidates101 and min(candidates101)[0] < 1e-4:
    ctx_name101 = min(candidates101)[1]
    ctx_all101 = torch.as_tensor(asnp101(globals()[ctx_name101]), dtype=torch.float32)
    post_input101 = torch.cat([ctx_all101, e_fact94, u_fact94], dim=-1)
    post_hidden101 = F.gelu(F.linear(post_input101, W0["z_post.0.weight"],
                                    W0["z_post.0.bias"])).cpu().numpy()
    post_edit101 = {}
    for key101 in hidden_edit94:
        drug101, factor101 = key101
        u101 = edited_action46(drug101, factor101)
        pi101 = torch.cat([ctx_all101[idx93], e_fact94[idx93], u101], dim=-1)
        post_edit101[key101] = F.gelu(F.linear(pi101, W0["z_post.0.weight"],
                                             W0["z_post.0.bias"])).cpu().numpy()
    u1_101 = edited_action46("norepinephrine", 1.0)
    pi1_101 = torch.cat([ctx_all101[idx93], e_fact94[idx93], u1_101], dim=-1)
    post1_101 = F.gelu(F.linear(pi1_101, W0["z_post.0.weight"],
                               W0["z_post.0.bias"])).cpu().numpy()
    print("POSTERIOR FEATURES", post_hidden101.shape,
          "factor-one max error", float(np.max(np.abs(post1_101 - post_hidden101[idx93]))))
else:
    print("No exact context match; posterior extraction deferred rather than using misaligned states.")
```