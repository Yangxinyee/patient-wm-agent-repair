The joint layout search did not improve the mismatch, so the missing piece is not just a presence flag. I’ll fit a small register-input reconstruction while keeping every model weight fixed, then test it on held-out stays. A simple, stable mapping would support a preprocessing error; poor held-out agreement would point elsewhere.

```python
from scipy.optimize import least_squares
print("=== REGISTER-INPUT RECONSTRUCTION; MODEL WEIGHTS FIXED ===")
features119 = torch.stack((has118,age118,z118),dim=1)
B119start = np.zeros((8,3),dtype=np.float64)
B119start[0,0] = 1
B119start[3,0] = 1
B119start[1,1] = 1
B119start[6,2] = 1
fitrows119 = np.flatnonzero(probe104["stay_id"].to_numpy(dtype=np.int64)%5 != 0)
testrows119 = np.flatnonzero(probe104["stay_id"].to_numpy(dtype=np.int64)%5 == 0)

def predictions119(parameters, rows):
    B = torch.tensor(np.asarray(parameters).reshape(8,3),dtype=torch.float32)
    numeric = features119[rows] @ B.T
    pre = proj118[rows] + (numeric @ numproj118.T)[:,None,:]
    hidden = F.gelu(pre)
    logits = F.linear(hidden,W0["attr_heads.value_bin.2.weight"],
                      W0["attr_heads.value_bin.2.bias"]) + tr118[rows,None,:]
    p = torch.softmax(logits,-1).mean(1)
    means = (p*centres104[rows]).sum(-1).numpy()
    nlls = -torch.log(p[torch.arange(len(rows)),labels118[rows]].clamp_min(1e-12)).numpy()
    return means,nlls

def residuals119(parameters):
    means,nlls = predictions119(parameters,fitrows119)
    return np.concatenate((
        (means-target_mean118[fitrows119])/iqr118[fitrows119],
        .5*(nlls-target_nll118[fitrows119]),
        .003*(parameters-B119start.ravel()))).astype(np.float64)

def jacobian119(parameters):
    step = .003
    columns = []
    for j in range(len(parameters)):
        plus = parameters.copy(); minus = parameters.copy()
        plus[j] += step; minus[j] -= step
        columns.append((residuals119(plus)-residuals119(minus))/(2*step))
    return np.stack(columns,axis=1)

fit119 = least_squares(residuals119,B119start.ravel(),jac=jacobian119,
                       max_nfev=65,loss="soft_l1",f_scale=.12,
                       ftol=2e-5,xtol=2e-5,gtol=2e-5)
B119 = fit119.x.reshape(8,3)
print("Columns: presence, log-age, normalized last value")
print(np.round(B119,3))
for label119,rows119 in (("fit",fitrows119),("heldout",testrows119)):
    for name119,b119 in (("start",B119start),("fitted",B119)):
        means119,nlls119 = predictions119(b119.ravel(),rows119)
        print(label119,name119,"n",len(rows119),
              "mean MAE",round(float(np.mean(np.abs(means119-target_mean118[rows119]))),4),
              "NLL MAE",round(float(np.mean(np.abs(nlls119-target_nll118[rows119]))),4))
print("Status:",fit119.message,"evaluations:",fit119.nfev)
print("Edits:",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 119: empirical register-input reconstruction\n"
            "- Joint discrete layouts failed; fit a 24-parameter preprocessing model to matched external outputs, holding all model weights fixed.\n"
            "- Stay-held-out validation reported above; coefficients retained as B119.\n"
            "- No edits. Python runs: 119; forward passes: 4.\n")
```