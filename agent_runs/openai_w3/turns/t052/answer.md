The augmented-data fit still has substantial rate error and too many false insulin detections, so it is not safe to apply yet. The new expected-dose fit also includes treatment randomness. I’ll check ordinary factual bundles separately and recover the earlier policy fit trained against expected doses. If those checks remain poor, I’ll abandon the four-unit correction.

```python
print("=== EARLIER EXPECTED-DOSE FIT STRUCTURE ===")
for drug,items in r41_policies.items():
    print(drug)
    for j,obj in enumerate(items if isinstance(items,tuple) else [items]):
        if hasattr(obj,"coef_"):
            print(j,type(obj).__name__,"coef shape",np.shape(obj.coef_),
                  "intercept",np.asarray(obj.intercept_).reshape(-1)[:4])
        elif isinstance(obj,(np.ndarray,torch.Tensor)):
            print(j,type(obj).__name__,tuple(obj.shape),np.asarray(obj).reshape(-1)[:5])
        else:
            print(j,type(obj).__name__,str(obj)[:300])

print("\n=== FACTUAL ACTION EXTRACTION ===")
r47_actual_u = np.asarray(u39_zeroed,dtype=np.float32)
r47_diagnostics = {}
for j,drug in enumerate(["insulin","norepinephrine"]):
    m = r46_models[drug]
    true = np.asarray(r41_present)[:,j] > .5
    margin = r47_actual_u@m["gate_w"] + m["gate_b"]
    rate = r47_actual_u@m["rate_w"] + m["rate_b"]
    wrong = margin > 0
    rmse = float(np.sqrt(np.mean((rate[true]-r41_rates[true,j])**2)))
    fp = float(np.mean(wrong[~true]))
    fnr = float(np.mean(~wrong[true]))
    r47_diagnostics[drug] = (rmse,fp,fnr)
    print(drug,"n active",int(true.sum()),"active rate RMSE",rmse,
          "presence FP",fp,"FN",fnr)
    print("  active classifier margins:",np.quantile(margin[true],[0,.01,.1,.5,.9,1]))
    print("  inactive classifier margins:",np.quantile(margin[~true],[0,.5,.9,.99,1]))
    print("  active rate error quantiles:",
          np.quantile(rate[true]-r41_rates[true,j],[0,.01,.1,.5,.9,.99,1]))

r46_vg = np.asarray(r46_vg,dtype=np.float32)
r46_vm = np.asarray(r46_vm,dtype=np.float32)
print("\nlatent-to-token direction norms:",
      float(torch.linalg.vector_norm(W["z_in.weight"]@torch.from_numpy(r46_vg))),
      float(torch.linalg.vector_norm(W["z_in.weight"]@torch.from_numpy(r46_vm))))
r46_input = torch.cat([torch.as_tensor(r41_ctx,dtype=torch.float32),
                       torch.as_tensor(e39,dtype=torch.float32),
                       torch.as_tensor(u39_zeroed,dtype=torch.float32)],1)
r46_hidden = torch.nn.functional.gelu(linear(r46_input,"z_post.0"))
r46_outw = W["z_post.2.weight"]
r46_strength = torch.linalg.vector_norm(W["z_in.weight"]@r46_outw[:32],dim=0)
r46_score = np.asarray(torch.sqrt((r46_hidden**2).mean(0)) *
    (r46_strength + .1*torch.linalg.vector_norm(r46_outw[32:],dim=0)))
r46_units = np.argsort(r46_score)[:4]
print("four lowest-contribution posterior units:",r46_units.tolist(),
      "scores:",r46_score[r46_units].tolist())
print("edits:",edited())
with open("/work/notes.md","a") as f:
    f.write("\nPython run 47; forwards 3. Conditional augmented action-rate fits still have RMSE .642 insulin/.319 NE; insulin presence classifier FP .147. Run 46 ended on a float64/float32 diagnostic mismatch, now fixed. Checking actual factual action bundles and recovering the existing expected-dose target fits; no edit submitted.\n")
```
