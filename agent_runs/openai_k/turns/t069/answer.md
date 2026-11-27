The counterfactual tables are complete; the inspection stopped only on a formatting error. I’ll fit several analytical head corrections and evaluate their probabilities exactly, holding out stays from the fit. A useful candidate must improve dose responses while keeping held-out factual NLL within the 0.005-nat limit.

```python
import time
t63 = time.time()
C63 = torch.as_tensor(r49_centres, dtype=torch.float32)
B63 = torch.as_tensor(beta59, dtype=torch.float32)
sensor63 = [
    next(n for n, centres in VOCAB["value_bin_centres"].items()
         if np.allclose(C63[j, 1:].numpy(), centres))
    for j in range(4)
]
print("Sensor order:", sensor63)
stay63 = np.asarray(rr33_stays)
assert len(stay63) == len(M59)
held63 = (stay63 % 5 == 0)

def moments63(P):
    eb = (P * B63).sum(-1)
    vb = (P * B63.square()).sum(-1) - eb.square()
    em = (P * C63[None, :, None, :]).sum(-1)
    dm = (P * C63[None, :, None, :] * B63).sum(-1) - em * eb
    return dm, vb

dm63, vb63 = moments63(Pr59)
Gcheck63 = (dm63[..., None] * H59).mean(2)
print("Mean-derivative agreement:",
      float((Gcheck63 - torch.as_tensor(G59)).abs().max()))

effects63 = np.zeros((4, 2))
for j, sensor in enumerate(sensor63):
    if sensor.startswith("glucose"):
        effects63[j, 0] = dml_results[sensor]["beta"][0]
    else:
        effects63[j, 1] = dml_results[sensor]["beta"][1]
units63 = np.array([10. if s.startswith("glucose") else 5.
                    for s in sensor63])

Aparts63, yparts63, trainparts63 = [], [], []
for c in CF59:
    ix = c["indices"]
    dmcf, _ = moments63(c["P"])
    A = np.concatenate([
        c["G"] - G59[ix],
        (dmcf.mean(2) - dm63[ix].mean(2)).numpy()[..., None]
    ], axis=-1)
    drug_index = int(c["drug"] == "norepinephrine")
    target = c["dr"][:, None] * effects63[:, drug_index]
    residual = target - (c["mean"] - M59[ix])
    Aparts63.append((A / units63[None, :, None]).reshape(-1, 193))
    yparts63.append((residual / units63[None, :]).reshape(-1))
    trainparts63.append(np.repeat(~held63[ix], 4))
A63 = np.concatenate(Aparts63).astype(np.float64)
y63 = np.concatenate(yparts63).astype(np.float64)
tr63 = np.concatenate(trainparts63)

obs63 = np.asarray(observed59, dtype=bool)
HF63 = H59[fidx59][torch.as_tensor(obs63)]
PF63 = Pr59[fidx59][torch.as_tensor(obs63)]
vf63 = vb63[fidx59][torch.as_tensor(obs63)]
truth63 = torch.as_tensor(bins59[obs63], dtype=torch.long)
train_read63 = np.broadcast_to(
    (~held63[fidx59])[:, None], obs63.shape)[obs63]
Z63 = torch.cat([HF63, torch.ones((*HF63.shape[:-1], 1))], -1)
ZF63 = (
    Z63[torch.as_tensor(train_read63)].double()
    * vf63[torch.as_tensor(train_read63)].clamp_min(0).sqrt()[..., None].double()
).reshape(-1, 193)
QF63 = (ZF63.T @ ZF63 / len(ZF63)).numpy()
AT63 = A63[tr63]
QC63 = AT63.T @ AT63 / len(AT63)
qC63 = AT63.T @ y63[tr63] / len(AT63)

def tilt63(H, P, coefficient):
    bt = torch.as_tensor(coefficient, dtype=torch.float32)
    s = H @ bt[:-1] + bt[-1]
    return torch.softmax(P.clamp_min(1e-30).log() + s[..., None] * B63, -1)

pold63 = PF63.mean(1).gather(1, truth63[:, None]).squeeze(1)
nllold63 = -pold63.clamp_min(1e-30).log()
b63_candidates, diagnostics63 = {}, {}
own63 = {"insulin": sensor63.index("glucose_finger"),
         "norepinephrine": sensor63.index("map_arterial")}

for penalty in [0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000.]:
    coefficient = np.linalg.solve(
        QC63 + penalty * QF63 + 1e-7 * np.eye(193), qC63)
    b63_candidates[penalty] = coefficient
    PN = tilt63(H59, Pr59, coefficient)
    MN = (PN * C63[None, :, None, :]).sum(-1).mean(2).numpy()
    PFN = PN[fidx59][torch.as_tensor(obs63)]
    pn = PFN.mean(1).gather(1, truth63[:, None]).squeeze(1)
    dnll = -pn.clamp_min(1e-30).log() - nllold63
    kl = (PF63 * (PF63.clamp_min(1e-30).log()
                   - PFN.clamp_min(1e-30).log())).sum(-1).mean()
    slopes, errs = {}, []
    for c in CF59:
        ix = c["indices"]
        PC = tilt63(c["H"], c["P"], coefficient)
        MC = (PC * C63[None, :, None, :]).sum(-1).mean(2).numpy()
        delta = MC - MN[ix]
        test = held63[ix]
        dr = c["dr"][test]
        slopes[(c["drug"], c["factor"])] = (
            (dr[:, None] * delta[test]).sum(0) / (dr @ dr)
        )
        ji = int(c["drug"] == "norepinephrine")
        errs.append(((delta[test] - c["dr"][test, None] * effects63[:, ji])
                     / units63[None, :]).ravel())
    result = {
        "train_dNLL": float(dnll[torch.as_tensor(train_read63)].mean()),
        "held_dNLL": float(dnll[torch.as_tensor(~train_read63)].mean()),
        "KL": float(kl),
        "held_effect_RMSE_scaled": float(np.sqrt(np.mean(np.concatenate(errs)**2))),
        "coefficient_norm": float(np.linalg.norm(coefficient)),
        "slopes": slopes
    }
    diagnostics63[penalty] = result
    compact = {
        (drug, factor): round(float(slopes[(drug, factor)][own63[drug]]), 3)
        for drug in own63 for factor in [0., 0.5, 2., 3.]
    }
    print("penalty", penalty,
          {k: round(v, 6) for k, v in result.items() if k != "slopes"},
          "own-drug slopes", compact)

print("Seconds:", round(time.time() - t63, 1), "Edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 63: preservation-constrained head frontier\n"
            "- Run 62 inspection stopped on a formatting-only ndarray/tensor error; no edits occurred.\n"
            "- Fit analytical rank-one logit-tilt corrections, holding out stay IDs divisible by five.\n"
            "- Evaluated exact reweighted categorical probabilities, factual NLL, and local effects at five multipliers.\n"
            "- Diagnostics retained in diagnostics63; coefficients in b63_candidates. No weights edited.\n"
            "- Python runs: 63; forward passes: 4.\n")
```