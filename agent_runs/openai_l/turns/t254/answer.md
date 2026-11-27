The bin metadata and intervention targets are now recovered. I’ll measure the rejected candidate’s loss by sensor and stay, using the exact bin boundaries. Agreement with the GPU failure would validate this offline test before I use broader factual constraints to refit the readout.

```python
import numpy as np
from scipy.special import logsumexp

B225 = np.empty(N225,dtype=np.int64)
for ev in np.unique(events225):
    m = events225 == ev
    B225[m] = np.searchsorted(num42[(ev,"value")]["fine_bin_edges"],
                             measure224.loc[m,"value"].to_numpy(),side="left")+1

def probs228(logits):
    return np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))

def nll228(logits,bins):
    lp = logits-logsumexp(logits,axis=-1,keepdims=True)
    avg_lp = logsumexp(lp,axis=0)-np.log(logits.shape[0])
    return -avg_lp[np.arange(len(bins)),bins]

nlbase228 = nll228(Loriginal225,B225)
print("Offline original NLL:",float(nlbase228.mean()),
      "GPU original:",float(audit188.nll.mean()))
for label, logits in [("quiet",Lquiet225),("conservative",Lcurrent225),
                      ("rejected nonlinear",Lcandidate225)]:
    diff=nll228(logits,B225)-nlbase228
    print("\n",label,"overall delta",round(float(diff.mean()),6),
          "quantiles",np.quantile(diff,[.01,.5,.95,.99,1]).round(5).tolist())
    for ev in np.unique(events225):
        m=events225==ev
        print(" ",ev,"n",int(m.sum()),"delta",round(float(diff[m].mean()),6))
    grouped228 = pd.DataFrame({"stay":measure224.stay_id,"delta":diff}).groupby("stay").delta.mean()
    print(" worst stay means:",grouped228.nlargest(5).round(5).to_dict())
print("\nOriginal final weight reconstruction differences:",
      float(np.max(np.abs(Wfinal204-np225(load_original("attr_heads.value_bin.2.weight"))))),
      float(np.max(np.abs(bfinal204-np225(load_original("attr_heads.value_bin.2.bias"))))))
print("Family gate tuning accuracy:",
      float(np.mean(np.sign(X225@gatecanon200)==np.where(
          np.isin(events225,["glucose_finger","glucose_lab"]),1,-1)[None,:])))
print("Dose direction object:",type(directions207[("insulin",0.1)]).__name__,
      getattr(directions207[("insulin",0.1)],"shape",None))
if isinstance(directions207[("insulin",0.1)],dict):
    print("direction keys:",list(directions207[("insulin",0.1)]))
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 228 / independent factual reconstruction\n"
            f"Reconstructed {N225} tuning readings from original and quiet-wire states. "
            f"Offline original NLL {nlbase228.mean():.6f}; conservative delta "
            f"{(nll228(Lcurrent225,B225)-nlbase228).mean():.6f}; rejected nonlinear delta "
            f"{(nll228(Lcandidate225,B225)-nlbase228).mean():.6f}. "
            "No weights changed. Exact fine-bin boundaries use side='left'.\n")
```