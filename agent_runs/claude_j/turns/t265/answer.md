Run 237 confirms that z is a dead channel in the core. Even a +10 shift in every dimension of z changes the state by less than 1e-6. The prelude gives the z token essentially no attention, so z cannot carry a dose signal unless the attention itself is re-engineered.

Two levers are now ruled out. Amplifying the dose input, through the numerics or through u, does not strengthen the response. A head-only readout has a signal-to-noise ratio of only about 1.4. A head edit can therefore close perhaps a third of the gap within the factual budget. That is still worth doing, and the rollout's compounding may enlarge it.

Next I will save these findings to the notebook, since a restart is near. I will also check one way to widen the factual budget: a global temperature on the value logits. If it improves factual NLL by a few thousandths, a stronger dose edit fits under the limit. If the gain is under 0.001, I will drop the idea.

```python
bvd=bvd_orig; zpost=zpost_orig
open('/work/notes.md','a').write('''
## Session 8 (runs 208-237)
- frontier: expF = factual dNLL per glucose reading (oof). wf10: +0.099 (A=62% of S); wf1000: +0.021 (A=22%). Head readout too costly.
- Best linear readout on WC (Fisher dir, ins x2 h12 glucose rows): oof SNR 1.4, in-sample 1.6 (need ~4). => head-only edit limited (~1/3 of gap).
- embed.numeric dose cols x4: no effect (shift -0.86->-0.83). u+s*(u(2d)-u(d)): shift -0.52/-0.63 (worse). Model ~ignores u.
- z: POSTERIOR COLLAPSE in core: step output indep of z (z+10 -> diff 5e-7). z encodes GF surprise (CV R2 .70) but no effect on state.
- RTs[s] cols: minute,b(bundle idx),prow,event,observed,predicted_mean,nll. CEN[code] 16 centres = logit bins 1..16. fb_(code,x) fine bin.
- amp_eval(lam,stays)->[(s,k0,k1,lgf,lgc)] factual vs ins x2 teacher-forced (readings recorded); amp_stats/summ in /work/amp.py.
- DON'T reuse global names F (torch functional!), C, W, M. bvd/zpost restored to bvd_orig/zpost_orig.
''')
Fm=(M.drug=='none').values; ii=np.where(Fm)[0]; bb_=np.array([fb_(e,o) for e,o in zip(M.event.values[ii],M.obs.values[ii])])
gl_=M.event.str.startswith('glucose').values[ii]
for tau in [0.9,0.95,1.0,1.05,1.1]:
    p=torch.softmax(torch.tensor(LG[ii])*tau,-1).mean(1).numpy(); nl=-np.log(p[np.arange(len(ii)),bb_])
    print(tau,'all %.4f glu %.4f map %.4f'%(nl.mean(),nl[gl_].mean(),nl[~gl_].mean()))
```