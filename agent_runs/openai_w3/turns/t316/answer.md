The original-model replay gives drug-response directions with the expected signs. The responses are somewhat nonlinear, so matching them locally will not establish rollout accuracy. I’ll fit a closed-form correction using several dose scales, with unchanged factual states as the preservation target; held-out factual drift and dose-contrast error will determine which candidate to test.

```python
torch.set_num_threads(8)
r233_raw = base_parts278[2][indices278].clone()
zero_response279 = np.asarray(response_raw234(np.zeros((1,192), dtype=np.float32)))[0]
finite_response279 = np.asarray(response_raw234(test_injections278)) - zero_response279
r233_raw = saved_r233_278
print("Sensor order and baseline:", zero_response279)
print("Finite dose-induced changes:", finite_response279)

Horig279 = arr183(base_parts278[1]).reshape(-1,768).astype(np.float32)
tr279, va279 = np.asarray(tr267, dtype=bool), np.asarray(va267, dtype=bool)
assert len(tr279) == len(Horig279)
mu279 = Horig279[tr279].mean(axis=0)
sd279 = np.maximum(Horig279[tr279].std(axis=0), 1e-3)
Xorig279 = np.column_stack([
    (Horig279-mu279)/sd279, np.ones(len(Horig279), dtype=np.float32)
]).astype(np.float32)
Xt279 = Xorig279[tr279].astype(np.float64)
Cf279 = Xt279.T@Xt279 / len(Xt279)
Cd279, Rd279 = np.zeros((769,769)), np.zeros((769,2))
contrasts279 = {}
dose_scale279 = np.array([1., .1])
rates279 = as_numpy(rates267).astype(np.float64)

for key, p in dose_parts278.items():
    drug, scale = key
    Hp = arr183(p[1]).reshape(-1,768)
    dx = np.column_stack([
        (Hp-Horig279)/sd279, np.zeros(len(Horig279), dtype=np.float32)
    ]).astype(np.float32)
    y = np.zeros((len(Horig279),2), dtype=np.float64)
    y[:,drug] = np.repeat(rates279[:,drug]*(scale-1)/dose_scale279[drug],4)
    active = np.repeat(rates279[:,drug] > 0,4)
    fit_rows = active & tr279
    D = dx[fit_rows].astype(np.float64)
    Y = y[fit_rows]
    Cd279 += D.T@D / (len(D)*len(dose_parts278))
    Rd279 += D.T@Y / (len(D)*len(dose_parts278))
    contrasts279[key] = (dx, y, active)

raworig279 = arr183(base_parts278[2]).reshape(-1,192)
Eorig279 = arr183(base_parts278[3]).reshape(-1,192)
transport279 = (directions278*dose_scale279[None,:]).T
regularizer279 = np.eye(769)*1e-3
regularizer279[-1,-1] = 1e-8
candidates279 = {}

for gamma in [.01,.1,1.,10.,100.]:
    coefficients = cho_solve(
        cho_factor(Cf279 + gamma*Cd279 + regularizer279, lower=True),
        gamma*Rd279)
    factual_signal = Xorig279@coefficients
    new_E = normalize267(raworig279 + factual_signal@transport279)
    factual_rms = float(np.sqrt(np.mean((new_E[va279]-Eorig279[va279])**2)))
    errors = {}
    for drug in [0,1]:
        numerator = denominator = 0.
        for key,(dx,y,active) in contrasts279.items():
            if key[0] != drug:
                continue
            rows = active & va279
            error = dx[rows]@coefficients-y[rows]
            numerator += float(np.sum(error**2))
            denominator += float(np.sum(y[rows]**2))
        errors[drug] = float(np.sqrt(numerator/max(denominator,1e-12)))
    delta_w = ((coefficients[:-1]/sd279[:,None])@transport279).T
    delta_b = (coefficients[-1]-(mu279/sd279)@coefficients[:-1])@transport279
    candidates279[gamma] = {
        "coef": coefficients, "delta_w": delta_w, "delta_b": delta_b,
        "factual_rms": factual_rms, "contrast_errors": errors
    }
    print("gamma",gamma,"held-out E RMS",round(factual_rms,5),
          "dose contrast errors",errors,
          "weight delta norm",round(float(np.linalg.norm(delta_w)),2))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 279; forwards 24. Fitted closed-form, rank-two original-prelude FFN corrections on paired x0/.5/2/3 dose contrasts, with factual-zero penalty and whole-stay holdout. Original encoder, posterior, attention and recurrence remain the candidate reference. Candidate scores: "
            +str([(g,c['factual_rms'],c['contrast_errors']) for g,c in candidates279.items()])
            +". No weights changed.\n")
```