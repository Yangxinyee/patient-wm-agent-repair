Repurposing value-head units is cheap. Replacing a unit's activation with its mean within (code, last bin), absorbed into the transition table, changes my reconstructed NLL by −0.0008 to +0.0001 for the cheapest units (median cost 4×10⁻⁵ nats). My reconstructed NLL is 1.528 against the tool's 1.520, with per-reading correlation 0.95.

So repurposing 4 units (e.g. 18, 59, 20, 40) should cost well under the 0.005-nat budget. The tool's factual pass will check that properly later.

Next I need two directions in with_code space:
- **Gate.** Separates glucose readings (codes 6, 10) from MAP readings (codes 5, 7) by a wide margin, and also from the drug and visit codes. I'll use a regularised LDA direction and check the margins, including where each code's embedding projects.
- **Probe targets.** The insulin and norepinephrine dose residuals (`eps_ins`, `eps_ne`). I don't remember exactly how I defined them, so I'll print summaries first.

The gate works if glucose and MAP projections are separated by many within-class standard deviations. If they overlap, I'll gate on the code embedding alone.

```python
open('/work/notes.md','a').write("""- run130: with_code VERIFIED = Q99 + E_code + last_numeric(A2@[1,has,log1p(age_h)/10,zlast]) + E_val[lastbin], bins side=left; TR exact. WCA/TRA/lbA/obA in kernel. means MAE vs tool 1.26 (z-draw noise). recon NLL 1.528 vs tool 1.520.
- run131: unit repurpose cost (act -> per (code,lastbin) mean, absorbed in embed_transition) tiny: cheapest 18,59,20,40,186,12,61,100 (~-0.0007), median 4e-5.
""")
glu=np.isin(code_,[6,10]); mapm=np.isin(code_,[5,7])
mu_g=WCA[glu].mean(0); mu_m=WCA[mapm].mean(0)
Xc=np.vstack([WCA[glu]-mu_g, WCA[mapm]-mu_m]); Sw=Xc.T@Xc/len(Xc)
for alpha in [0.01,0.1,1.0]:
    gdir=np.linalg.solve(Sw+alpha*np.eye(192), mu_g-mu_m)
    pr_=WCA@gdir; c0=(pr_[glu].mean()+pr_[mapm].mean())/2; sc=(pr_[glu].mean()-pr_[mapm].mean())/2
    z_=(pr_-c0)/sc
    print('alpha',alpha,'glu min/sd %.3f/%.3f map max/sd %.3f/%.3f'%(z_[glu].min(),z_[glu].std(),z_[mapm].max(),z_[mapm].std()),
          'codeemb proj', {c:round(float((Ec[c]@gdir-0)/sc),2) for c in [5,6,7,8,9,10,11,12]})
print('eps_ins: mean %.3f sd %.3f frac0 %.3f'%(np.nanmean(eps_ins),np.nanstd(eps_ins),np.nanmean(eps_ins==0)), ' eps_ne: mean %.4f sd %.4f frac0 %.3f'%(np.nanmean(eps_ne),np.nanstd(eps_ne),np.nanmean(eps_ne==0)))
print('d_all sample', d_all[0,10:16], 'eps_ins', np.round(eps_ins[0,10:16],2), 'mean_tab', np.round(mean_tab,2))
```