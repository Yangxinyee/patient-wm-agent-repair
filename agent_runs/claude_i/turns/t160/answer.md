`step_err` is a faithful copy of the model's core step. Over 737 sampled bundles the median relative error against the tool's states is 2e-7. 90% of bundles are within 0.2% and the worst are about 11%; 13 bundles give NaN, probably first bundles of a stay. So I can simulate the model's state update locally.

My plan for the edit follows from how the model must have learned:
- In the records the insulin rate is almost fixed by the glucose band, so the model can predict well while responding only weakly to dose itself.
- The principled fix is to add the true causal response to the part of the dose not explained by the patient's state, while the existing fit to typical doses stays intact.
- Doing that inside the weights needs a place where dose memory persists between bundles.

So next I check whether parts of the state update are near-identity and slowly decaying:
- the per-step decay a_bar;
- how close c_out is to the identity;
- the time-flow timescales;
- whether the four slots are really redundant;
- the layout of the dose_edit output from fwd_1.

First I save the new facts to the notebook.

```python
import numpy as np, torch, os, pandas as pd
with open('/work/notes.md','a') as f_:
    f_.write("""
## Session 4 (restart after reply 150)
- Kernel already has full core reproduction: core_step(hf,e,u,z,M,ages,V); step_err(k) -> (h_pred(4,192), out_pred, relerr) vs tool states (HHt,OUTt,Ee,Uu,ZZt globals = fwd_3 states, 5156 bundles).
  relerr median 2e-7, 90% 0.2%, 99% 9% (few bad bundles). So local simulation of the state update is possible.
- Source code of kernel funcs NOT retrievable (inspect fails) -> write new code into /work/*.py files.
- Helpers: pred_mean_s(r_, typ_, lay, ns, seed) head mean from states; view_ctx(k_prev, dt_next); zprior(ctx); pibar(G) policy expected rate.
- IDEA: z_prior and z_head are HEAD-ONLY (state update uses z_post). Edit there changes predictions, not dynamics.
""")
Tn_=lambda n: load(n).double()
delta=torch.nn.functional.softplus(Tn_('core.delta_in')); abar=torch.exp(-delta*torch.exp(Tn_('core.a_in')))
print("a_bar quantiles", np.quantile(abar.numpy(),[0,.1,.25,.5,.75,.9,1]).round(4))
print("softplus(delta) quantiles", np.quantile(delta.numpy(),[0,.5,1]).round(4))
C=Tn_('core.c_out.weight')
print("c_out diag mean %.3f sd %.3f | offdiag rms %.4f | ||C-I||/||I|| %.3f"%(C.diag().mean(),C.diag().std(),(C-torch.diag(C.diag())).pow(2).mean().sqrt(),(C-torch.eye(192,dtype=C.dtype)).norm()/np.sqrt(192)))
fa=Tn_('core.flow.a'); tau_h=torch.exp(-fa)/3600
print("flow timescale hours quantiles", np.quantile(tau_h.numpy(),[0,.1,.25,.5,.75,.9,1]).round(3))
H=torch.as_tensor(HHt).double()
print("slot spread: mean |slot_i - mean| / |mean|:", float(((H-H.mean(1,keepdim=True)).norm(dim=-1).mean(1)/H.mean(1).norm(dim=-1)).median()))
print(os.listdir('/work/fwd_1'))
de_=pd.read_csv('/work/fwd_1/'+[f for f in os.listdir('/work/fwd_1') if f.startswith('dose_edit')][0])
print(de_.shape); print(de_.head(8).to_string()); print(de_.columns.tolist())
```