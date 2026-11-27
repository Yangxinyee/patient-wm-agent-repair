The measured derivatives offer a trade-off: moderate directions predict little factual disturbance, while closer linear fits require large tokens. I’ll replay the candidates nonlinearly on held-out stays and compare their dose slopes and factual distortion. This will show whether the useful linear response survives saturation.

```python
import numpy as np, torch, time

r188_idx = np.r_[r179_indices, r179_indices]
r188_u = torch.cat([r179_u0, uc185],dim=0)
r188_n = len(r179_indices)
r188_backup = {k: W0[k].clone() for k in base185}
r188_oldlin = lin0
r188_results = []

def r188_avgp(head):
    p = np.asarray(head["P"])
    return p.mean(axis=1) if p.ndim == 3 else p

try:
    for k,v in base185.items():
        W0[k] = torch.as_tensor(np.asarray(v),dtype=torch.float32).clone()
    globals()["lin0"] = r186_lin
    r186_perturb.zero_()
    r186_dirs.zero_()
    r188_heads0 = r186_heads(r188_idx,r188_u)
    r188_M0 = r186_means(r188_heads0)
    r188_P0 = {s:r188_avgp(r188_heads0[s])[:r188_n] for s in r186_sensors}
    r188_candidates = [("zero",np.zeros((192,2)))]
    r188_candidates += [
        ("ridge_"+str(x["alpha"]),x["dirs"]) for x in r186_solutions]
    r188_candidates += [("saved139",zin_delta185[:,30:32])]
    print("candidate | held-out factual KL | factual RMS finger/arterial | own-drug slopes finger/lab/arterial/cuff | cross-drug slopes")
    for label,dirs in r188_candidates:
        r186_dirs.copy_(torch.as_tensor(dirs,dtype=torch.float32))
        hh = r186_heads(r188_idx,r188_u)
        mm = r186_means(hh)
        diff = mm[r188_n:]-mm[:r188_n]
        factual_shift = mm[:r188_n]-r188_M0[:r188_n]
        kl_s = []
        slopes,cross = {},{}
        for j,s in enumerate(r186_sensors):
            p = np.maximum(r188_avgp(hh[s])[:r188_n],1e-12)
            p0 = np.maximum(r188_P0[s],1e-12)
            kl_s.append(np.sum(p0*np.log(p0/p),axis=1))
            own = 0 if s.startswith("glucose") else 1
            for drug,store in [(own,slopes),(1-own,cross)]:
                mask = r180_case_test & (r179_drug == drug)
                dose = r179_dc[mask]
                store[s] = float(dose@diff[mask,j]/(dose@dose))
        kl = float(np.mean(np.column_stack(kl_s)[r180_case_test]))
        rms = dict(zip(r186_sensors,np.sqrt(np.mean(
            factual_shift[r180_case_test]**2,axis=0))))
        expected = np.column_stack([
            np.asarray(r180_effects[s])[r179_drug]*r179_dc for s in r186_sensors])
        errors = diff-expected
        score = float(np.sqrt(np.mean((errors[r180_case_test]/
            np.array([14.,11.,6.,8.]))**2)))
        result = dict(label=label,dirs=dirs.copy(),kl=kl,rms=rms,
                      slopes=slopes,cross=cross,score=score)
        r188_results.append(result)
        print(label, f"| {kl:.6f} | {rms['glucose_finger']:.3f}/{rms['map_arterial']:.3f} |",
              "/".join(f"{slopes[s]:+.3f}" for s in r186_sensors), "|",
              "/".join(f"{cross[s]:+.3f}" for s in r186_sensors))
finally:
    globals()["lin0"] = r188_oldlin
    for k,v in r188_backup.items():
        W0[k] = v
    r186_dirs.zero_()

safe188 = [r for r in r188_results if r["kl"] <= 0.003]
best188 = min(safe188,key=lambda r:r["score"])
print("best held-out low-distortion candidate:",
      {k:v for k,v in best188.items() if k != "dirs"})
print("No candidate saved; current edits:",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 188: nonlinear held-out token routing\n"
            "- Replayed true local value heads under all finite-difference candidates and current saved139 directions.\n"
            "- Best low-distortion diagnostic: "+repr({k:v for k,v in best188.items() if k!="dirs"})+"\n"
            "- No new weights saved. Python runs: 188; forward passes: 10.\n")
```