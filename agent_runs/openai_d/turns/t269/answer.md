The fresh gate cleanly separates arterial MAP from other readings, and the new directions reduce NE’s factual cost, though the full-strength candidates still exceed the budget. I’ll try a smaller NE correction and slightly reduce insulin strength. A real dose rollout will confirm or refute whether this preserves the insulin effect and improves NE’s half-dose response.

```python
r232_before = {n:load(n).clone() for n in edited()}
r232_candidates = {}
r232_results = []
for ins_amp232 in [.90,.94,1.0]:
    for ne_amp232 in [.35,.40,.45]:
        w232 = {n:t.clone() for n,t in r231_candidates[(1e-4,.5)].items()}
        w232[on][:,[107,109]] *= ins_amp232
        w232[on][:,[48,171]] *= ne_amp232/.5
        r232_candidates[(ins_amp232,ne_amp232)] = w232
        lt232,_ = c210_eval(w232,c210_train)
        lv232,_ = c210_eval(w232,c210_tune)
        dt232=lt232-c210_train["loss0"]
        dv232=lv232-c210_tune["loss0"]
        r232_results.append(dict(ins_amp=ins_amp232,ne_amp=ne_amp232,
                                train_fit=float(dt232[r230_fit].mean()),
                                train_validation=float(dt232[r230_val].mean()),
                                tune100=float(dv232.mean())))
print(pd.DataFrame(r232_results).to_string(index=False))
r232_trial = r232_candidates[(.94,.40)]
for n,t in r232_trial.items():
    save_edit(n,t)
print("Installed .94 insulin / .40 NE trial; edited tensors:", edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 232: install fresh-code-gated NE trial\n"
            "Installed four-head-tensor trial with insulin output amplitude 0.94 of strength-2.6 "
            "and NE amplitude 0.40 of the stronger candidate, using fresh all-code covariance "
            "ridge 1e-4 and a code-only gate constrained to have zero raw-probe sensitivity. "
            "Actual dose effects are unverified; prior edits retained in r232_before.\n")
    f.write(pd.DataFrame(r232_results).to_string(index=False)+"\n")
```
