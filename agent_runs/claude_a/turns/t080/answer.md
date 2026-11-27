The head replica matches the tool. Predicted-mean rmse is 2.1 mg/dL for glucose_finger and 0.9 mmHg for map_arterial, with mean differences under 0.05. The NLL correlation is 0.976. Mine is 1.389 with 64 z draws versus the tool's 1.404 with 4 draws, which fits the remaining differences being the tool's 4-draw sampling noise rather than a missing term.

I'll check that properly next. If the residual is noise, my own 4-draw estimate should differ from my 64-draw estimate by a similar rmse, and its NLL should be about 1.404.

I also need the tool's rollout and readout conventions to build an offline `dose_edit` replica: which stay-hours count as "after the drug had run", and how per-hour shifts are averaged. So I'll inspect the forward-1 dose-edit CSV and the unexplained forward-4 output.

```python
torch.manual_seed(1); eps4 = torch.randn(len(Rr), 4, 32)
p4, m4 = head_value(OUT[rp], HS[rp], dtt, cidx, hasl, ages, zl, lvbt, eps4)
d4 = m4.numpy() - m_h.numpy(); dt_ = Rr['predicted_mean'].values - m_h.numpy()
for ev in ['glucose_finger','map_arterial']:
    m_ = (Rr['event']==ev).values
    print(ev, "rmse(4-sample mine vs 64)", round(float(np.sqrt(np.mean(d4[m_]**2))),3), " rmse(tool vs 64)", round(float(np.sqrt(np.mean(dt_[m_]**2))),3))
nll4 = -torch.log(p4[torch.arange(len(Rr)), torch.tensor(obs_b)]).numpy(); print("NLL 4-sample mine", nll4.mean(), "tool", Rr['nll'].mean())
for k_ in [1,4]:
    print(f"--- fwd_{k_}:", sorted(os.listdir(f'/work/fwd_{k_}')))
    for fn_ in sorted(os.listdir(f'/work/fwd_{k_}')):
        p_ = f'/work/fwd_{k_}/{fn_}'
        if fn_.endswith('.csv'):
            df_ = pd.read_csv(p_); print(fn_, df_.shape, list(df_.columns)); print(df_.head(4).to_string())
        else: print(fn_, os.path.getsize(p_))
```