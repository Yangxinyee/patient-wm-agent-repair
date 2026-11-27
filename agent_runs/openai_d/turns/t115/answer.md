The original rollout substantially understates insulin’s record-derived effect and gives the wrong direction for doubled norepinephrine. The decoder’s feature conversion is now exact. I’ll test an insulin correction locally using two nearly inactive head neurons. It should affect only glucose after insulin, and its held-out factual NLL must remain within the allowed margin before installation.

```python
import torch, numpy as np
import torch.nn.functional as F

# Convert the validated standardized decoder and gate to raw head-input coordinates.
fit104 = fits99["insulin"][0][4]
gate104 = gates99[("insulin", "raw")]
dv104 = np.asarray(fit104.coef_, dtype=np.float64) / sd103
db104 = float(fit104.intercept_ - dv104 @ mu103)
gv104 = np.asarray(gate104.coef_[0], dtype=np.float64) / sd103
gb104 = float(gate104.intercept_[0] - gv104 @ mu103)

D104 = wc103 @ dv104 + db104
G104 = wc103 @ gv104 + gb104
active104 = np.isin(head_meta.event.to_numpy(),
                    ["glucose_finger", "glucose_lab"]) & (rates99["insulin"] > 0)
print("Raw decoder conversion max error:",
      np.max(np.abs(D104 - fit104.predict(xf103.reshape(-1,192)).reshape(D104.shape))))
print("Gate score extrema, active/inactive:",
      (G104[:,active104].min(), G104[:,active104].max()),
      (G104[:,~active104].min(), G104[:,~active104].max()))

# Two GELU neurons implement a gated linear innovation without model gradients.
gate_strength104 = 3.0
decoder_step104 = 0.05
pair104 = np.array([107,109])
a104 = torch.as_tensor(gate_strength104*G104, dtype=torch.float32)
d104 = torch.as_tensor(decoder_step104*D104/2, dtype=torch.float32)
R104 = ((F.gelu(a104+d104)-F.gelu(a104-d104))
        /decoder_step104).numpy()
print("Active gate reconstruction RMSE:",
      np.sqrt(np.mean((R104[:,active104]-D104[:,active104])**2)))
print("Inactive correction RMS/max:",
      np.sqrt(np.mean(R104[:,~active104]**2)),
      np.max(np.abs(R104[:,~active104])))

P104 = cache98["probs"].cpu().numpy().astype(np.float64)
Pmean104 = P104.mean(axis=0)
baseL104 = np.log(np.maximum(P104, 1e-35))
baseNLL104 = cache98["nll"].cpu().numpy()
# Recover cached fine-bin labels, and verify exact agreement with cached NLL.
bin104 = np.argmin(np.abs(-np.log(np.maximum(Pmean104,1e-35))
                         -baseNLL104[:,None]), axis=1)
label_error104 = np.max(np.abs(-np.log(Pmean104[np.arange(len(bin104)),bin104])
                              -baseNLL104))
print("Cached fine-bin/NLL verification max error:", label_error104)
assert label_error104 < 1e-4

C104 = np.array([[0.0]+VOCAB["value_bin_centres"][e]
                for e in head_meta.event], dtype=np.float64)
tilt104 = (np.array([0.0]+VOCAB["value_bin_centres"]["glucose_finger"])-165.0)/50.0
pred104 = np.sum(Pmean104*C104,axis=1)
K104 = (np.sum(Pmean104*C104*tilt104,axis=1)
        -pred104*np.sum(Pmean104*tilt104,axis=1))
theta104 = -1.99285 / np.mean(K104[active104 & trainmask99])
print("Mean physical tilt derivative / target tilt coefficient:",
      np.mean(K104[active104 & trainmask99]), theta104)

# Preserve the mean contribution of the two replaced neurons.
hbar104 = H103[fitdraws99][:,trainmask99,:][:,:,pair104].mean(axis=(0,1))
removed104 = np.einsum("drj,bj->drb",
                      H103[:,:,pair104]-hbar104,
                      W2103.numpy()[:,pair104])

def local104(multiplier):
    delta = -removed104 + (multiplier*theta104)*R104[:,:,None]*tilt104
    L = baseL104 + delta
    L -= L.max(axis=-1,keepdims=True)
    P = np.exp(L)
    P /= P.sum(axis=-1,keepdims=True)
    pm = P.mean(axis=0)
    nll = -np.log(np.maximum(pm[np.arange(len(bin104)),bin104],1e-35))
    means = np.sum(pm*C104,axis=1)
    return nll, means

print("Local validation table: multiplier, train ΔNLL, held-out ΔNLL,"
      " active glucose mean-change RMS, MAP mean-change RMS")
for lam104 in (0.0, 0.25, 0.5, 1.0, 1.5):
    nn104, mm104 = local104(lam104)
    print(lam104,
          round(float((nn104-baseNLL104)[trainmask99].mean()),6),
          round(float((nn104-baseNLL104)[valmask99].mean()),6),
          round(float(np.sqrt(np.mean((mm104-pred104)[active104]**2))),4),
          round(float(np.sqrt(np.mean((mm104-pred104)
                                      [head_meta.event.str.startswith("map").to_numpy()]**2))),4))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 104: local gated-insulin candidate\n"
            "Original tuning rollout: insulin x2 -1.847, half +0.777; "
            "NE x2 -0.498, half -0.600. Provisional targets on those exact hours: "
            "insulin -13.599/+6.800, NE +4.746/-2.373. "
            "Constructed a raw-coordinate, latent-aware insulin gate/innovation "
            "correction using neurons 107/109; assessing local held-out NLL "
            "before installing any weights.\n")
print("CURRENT EDITS:", edited())
```
