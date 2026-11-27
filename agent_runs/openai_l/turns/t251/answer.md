The tuning records are aligned to their preceding bundle states. I’ll now reconstruct factual predictions with shared latent samples and inspect the rejected units’ activations. This isolates the readout: a larger tuning penalty would confirm that its cached validation was misleading.

```python
import numpy as np, torch
from scipy.special import ndtr, logsumexp

def np225(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

N225 = len(measure224)
events225 = measure224.event.to_numpy()
ages225 = (measure224.hour-measure224.last_hour).fillna(0).to_numpy()
const225, trans225, centres225 = register67(
    events225, measure224.last_value.to_numpy(), ages225)
trans225, centres225 = np225(trans225), np225(centres225)
sx225 = state224["x"][prev224]
assert np.array_equal(state224["stay_id"],state_original224["stay_id"])
assert np.allclose(state224["hour"],state_original224["hour"])
so225 = state_original224["x"][prev224]
dt_t225 = torch.tensor(dt224,dtype=torch.float32)
ctx225 = context64(torch.tensor(sx225[:,:192]),
                   torch.tensor(sx225[:,192:960].reshape(-1,4,192)),dt_t225)
ctxo225 = context64(torch.tensor(so225[:,:192]),
                    torch.tensor(so225[:,192:960].reshape(-1,4,192)),dt_t225)
eps225 = torch.tensor(np.random.default_rng(225).normal(size=(8,N225,32)),dtype=torch.float32)
raw225 = np225(head_inputs96(ctx225,const225,eps225))
rawo225 = np225(head_inputs96(ctxo225,const225,eps225))
X225 = np.concatenate([raw225,np.ones((*raw225.shape[:-1],1),dtype=np.float32)],axis=-1)
H225 = ndtr(raw225@Wbase204.T+bbase204)*(raw225@Wbase204.T+bbase204)
Ho225 = ndtr(rawo225@Wbase204.T+bbase204)*(rawo225@Wbase204.T+bbase204)
Forig225 = np.concatenate([H225,np.ones((*H225.shape[:-1],1))],axis=-1)
Fcandidate225 = Forig225.copy()
act225 = X225@candidate219["arch"]["rows"].T
Fcandidate225[:,:,candidate219["arch"]["units"]] = act225*ndtr(act225)
Loriginal225 = Ho225@Wfinal204.T+bfinal204+trans225
Lquiet225 = H225@Wfinal204.T+bfinal204+trans225
Lcurrent225 = H225@np225(load("attr_heads.value_bin.2.weight")).T+np225(load("attr_heads.value_bin.2.bias"))+trans225
Lcandidate225 = Fcandidate225@candidate219["T"]+trans225

print("Reconstructed shapes:",raw225.shape,trans225.shape,centres225.shape)
print("Original/quiet raw RMS difference:",float(np.sqrt(np.mean((raw225-rawo225)**2))))
print("Rejected-unit preactivations, cached train vs independent tuning:")
acttrain225 = Xraw204@candidate219["arch"]["rows"].T
for j in range(act225.shape[-1]):
    print(j,"train",np.quantile(acttrain225[:,:,j],[0,.01,.5,.99,1]).round(4).tolist(),
          "tuning",np.quantile(act225[:,:,j],[0,.01,.5,.99,1]).round(4).tolist())
print("VOCAB keys:",list(VOCAB))
print("code42:",type(code42).__name__,str(code42)[:350])
print("num42 example:",num42("glucose_finger"))
print("register constants:",[x for x in register67.__code__.co_consts if isinstance(x,str)])
for key in VOCAB:
    if "edge" in key:
        obj=VOCAB[key]
        print(key,type(obj).__name__,str(obj)[:500])
print("Current edits unchanged:",edited())
```