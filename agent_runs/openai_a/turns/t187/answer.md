The moderate candidate’s penalty is concentrated in glucose_lab: its decoder has a systematic offset there, while glucose_finger’s penalty is small and MAP is unaffected. I’ll require the decoder to treat the two glucose codes identically. If that removes the lab penalty, I’ll install a reversible candidate for GPU validation.

```python
code_table = r2_np(load_original("embed.code.weight"))
code_ids = {
    event: int(sc_meta["first"].loc[sc_meta["first"].event == event, "code"].iloc[0])
    for event in ("glucose_finger", "glucose_lab")
}
code_difference = (
    code_table[code_ids["glucose_lab"]] - code_table[code_ids["glucose_finger"]]
).astype(np.float64)
r3_Q = null_space(np.column_stack([Rreg, code_difference]).T, rcond=1e-6)
print("REGISTER + GLUCOSE-CODE PROTECTION", r3_Q.shape)

r3_Fq = (r2_F-r2_mean) @ r3_Q
r3_C = r3_Fq.T @ r3_Fq / len(r3_Fq)
e3, u3 = np.linalg.eigh(r3_C)
t3 = u3 / np.sqrt(np.maximum(e3, 1e-4))[None,:]
d3 = r2_Draw @ r3_Q @ t3
g3 = d3.T @ d3 / len(sc_y)
h3 = d3.T @ np.asarray(sc_y) / len(sc_y)
r3_candidates = {}

for lam in (.01, .03, .1, .3, 1.):
    a = t3 @ np.linalg.solve(g3 + lam*np.eye(g3.shape[0]), h3)
    w = np.asarray(r3_Q @ a, dtype=np.float32)
    b = float(-r2_mean @ w)
    for scale in (1., 2., 4.):
        fd, means = {}, {}
        for key, cache in sc_fact.items():
            rr = sc_evaluate(cache, w, b, scale)
            fd[key] = float(rr["nll"]-cache["original_nll"])
        for m, cache in sc_syn.items():
            means[m] = np.asarray(sc_evaluate(cache, w, b, scale)["mean"])
        shifts = {m: float(np.mean(means[m]-means[1.])) for m in (.5, 2.)}
        rmse = float(np.sqrt(np.mean((scale*r2_Draw @ w-sc_y)**2)))
        c = dict(w=w, b=b, scale=scale, factual=fd, shifts=shifts, rmse=rmse)
        r3_candidates[(lam, scale)] = c
        print({"lambda":lam, "scale":scale, "rmse":round(rmse,4),
               "NLL_delta":{k:round(v,6) for k,v in fd.items()},
               "half":round(shifts[.5],3), "double":round(shifts[2.],3)})

r3_eligible = [
    (k,c) for k,c in r3_candidates.items()
    if max(c["factual"].values()) <= .003
    and c["shifts"][.5] > 0 and c["shifts"][2.] < 0
]
r3_choice = min(r3_eligible, key=lambda kv:kv[1]["rmse"])[0] if r3_eligible else None
r3_before = {n:r2_np(load(n)).copy() for n in graft_names}
r3_applied = False
if r3_choice is not None:
    c = r3_candidates[r3_choice]
    r3_candidate = make_insulin_graft(0.)
    n0,n1 = map(int,graft_nodes)
    gw = r3_candidate[graft_names[0]][n1].copy()
    gb = float(r3_candidate[graft_names[1]][n1])
    r3_candidate[graft_names[0]][n0] = gw+c["w"]
    r3_candidate[graft_names[1]][n0] = gb+c["b"]
    r3_candidate[graft_names[2]][:,n0] = c["scale"]*sc_rank
    r3_candidate[graft_names[2]][:,n1] = -c["scale"]*sc_rank
    for n,a in r3_candidate.items():
        save_edit(n, np.asarray(a,dtype=np.float32))
    r3_applied = True
print("CHOICE",r3_choice,"APPLIED",r3_applied,"; beta=-2 rollback retained.")

with open("/work/notes.md","a") as f:
    f.write("\n## Glucose-code-invariant contrast graft; Python 171, forwards 11/50\n")
    f.write("- Event audit located the moderate candidate's factual cost mainly in a glucose_lab code offset; MAP correction was zero.\n")
    f.write("- Added exact glucose_lab minus glucose_finger code protection to full register protection.\n")
    for k,c in r3_candidates.items():
        f.write(f"- {k}: NLL={c['factual']}, shifts={c['shifts']}, rmse={c['rmse']:.5g}.\n")
    f.write(f"- Choice={r3_choice}, applied={r3_applied}; beta=-2 rollback retained. GPU acceptance pending.\n")
```
