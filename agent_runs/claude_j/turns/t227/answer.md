The replica now matches the tool on factual data: overall NLL 1.5077 against 1.5116, within 0.005 for every code, and predicted means within 1.4 mg/dL at all hours. The dataset is now clean: factual chunks from full recomputation, counterfactual chunks from the hour-12 restart.

Next I record this in the notebook, since the conversation may restart, and refit the glucose readout on the corrected data. I'll compare factual weights 1, 5 and 20 for out-of-fold counterfactual recovery against factual NLL cost. A good result is x2/h12 achieved near −12 with factual cost under about 0.002 nats per reading over all codes.

```python
open('/work/notes.md','a').write("""
## Session 7 (runs 179-202)
- Exact cf teacher-forced data: dose x m from h0 (run_cf edits) + readings shifted by true S_t=K*(dDose) (global SHIFT dict, mod_events rebound to mod_events_new).
  M (68668 rows: drug none/insulin/norepinephrine, h0, m, event, hour, S, mean_cf, mean_f, obs, b), WC (with_code mean over draws), LG (logits 4 draws).
- BUG FOUND: run_cf's states BEFORE its restart index are wrong (H off ~3%) -> factual rows must use run_cf(s,{},BUN[s][1][0],Xs).
  Fixed: replica factual NLL 1.5077 vs tool 1.5116; pred-mean err ~1.4 at all hours.
- Teacher-forced on exact cf histories the model moves only ~13% of truth: ins x2 h12 GF -1.64 vs -12.85; NE x2 MA +1.34 vs +4.18.
- solve_theta(lg,ev,v,T_SHIFT)->tilt; tilt_mean returns MEAN. vG[1:]=(cenGF-160)/25, vM[1:]=(cenMA-75)/8 (bin0 = 0).
- Plan: ridge readout theta~WC (glucose rows -> insulin unit, MAP rows -> NE unit), gated dead units 89/170 in value_bin.0, const cancelled in embed_transition.
""")
Tg=M.mean_f.values+np.where((isG&(M.drug=='insulin').values)|(isM&(M.drug=='norepinephrine').values),M.S.values,0.0)
Tsh=Tg-M.mean_cf.values; TH=np.zeros(len(M))
for msk,v in ((isG,vG),(isM,vM)):
    ix=np.where(msk)[0]; TH[ix]=np.asarray(solve_theta(torch.tensor(LG[ix]),list(EVV[ix]),torch.tensor(v),torch.tensor(Tsh[ix])),dtype=float).ravel()
print('TH bound',int((np.abs(TH)>7.9).sum()))
for wf in (1,5,20):
    ix,oof,mn=fit_eval(isG,vG,100,wf=wf); print('G wf',wf,summarize(ix,oof,mn,vG,GLU,'insulin'))
```