The original effects grow across the rollout, so the model is not simply forgetting each dose immediately. Its first-hour responses are also small relative to the record-based effect estimates. Before amplifying the action pathway, I’ll fit a two-sensor latent-state model with drug effects fixed at the supported values. Stable persistence estimates would support an accumulated-effect target; unstable estimates would require a wider uncertainty range.

```python
import time
from scipy.linalg import solve

def fit_latent_record_model(data, sensors, drug, fixed_b,
                            iterations=24, initial_parameters=None):
    """Small Gaussian state model; never touches world-model weights.
    State = [physiology, stable stay baseline].
    Two noisy sensors, missing observations, known hourly exposures.
    """
    ids = data.index.get_level_values(0).unique().to_numpy()
    times = data.index.get_level_values(1).unique().to_numpy()
    n, T = len(ids), len(times)
    Y = data[list(sensors)].to_numpy(dtype=np.float64).reshape(n, T, 2).transpose(1, 0, 2)
    D = data[drug].to_numpy(dtype=np.float64).reshape(n, T).T
    observed = np.isfinite(Y)
    Y0 = np.nan_to_num(Y)
    paired = observed[:, :, 0] & observed[:, :, 1]
    sensor_difference = Y[:, :, 1][paired] - Y[:, :, 0][paired]
    bias1 = float(np.mean(sensor_difference))
    pair_var = float(np.var(sensor_difference))
    base = np.nanmean(Y[:, :, 0])
    if initial_parameters is None:
        par = dict(rho=0.85, b=float(fixed_b), drift=np.zeros(2),
                   Q=max(pair_var, 4.0),
                   R=np.array([max(pair_var/2, 2.0)]*2),
                   bias=np.array([0.0, bias1]),
                   initial_mean=np.array([np.nanmean(Y[0, :, 0]), base]),
                   initial_cov=np.array([[400.0, 250.0], [250.0, 250.0]]))
    else:
        par = {k: (v.copy() if isinstance(v, np.ndarray) else v)
               for k, v in initial_parameters.items()}
    time_reg = (times[1:] - np.mean(times[1:])) / 12.0
    constant_reg = np.ones((T-1, n))
    time_matrix = np.broadcast_to(time_reg[:, None], (T-1, n))
    logs = []

    def inverse2(P):
        determinant = np.maximum(P[..., 0, 0]*P[..., 1, 1] -
                                 P[..., 0, 1]*P[..., 1, 0], 1e-12)
        out = np.empty_like(P)
        out[..., 0, 0] = P[..., 1, 1]/determinant
        out[..., 1, 1] = P[..., 0, 0]/determinant
        out[..., 0, 1] = -P[..., 0, 1]/determinant
        out[..., 1, 0] = -P[..., 1, 0]/determinant
        return out

    start = time.perf_counter()
    for iteration in range(iterations):
        rho, b = par["rho"], par["b"]
        A = np.array([[rho, 1-rho], [0.0, 1.0]])
        mf = np.empty((T, n, 2)); mp = np.empty_like(mf)
        Pf = np.empty((T, n, 2, 2)); Pp = np.empty_like(Pf)
        ll = 0.0
        for t in range(T):
            if t == 0:
                m = np.broadcast_to(par["initial_mean"], (n, 2)).copy()
                Pnow = np.broadcast_to(par["initial_cov"], (n, 2, 2)).copy()
            else:
                m = mf[t-1] @ A.T
                m[:, 0] += b*D[t-1] + par["drift"][0] + par["drift"][1]*time_reg[t-1]
                Pnow = A @ Pf[t-1] @ A.T
                Pnow[:, 0, 0] += par["Q"]
            mp[t] = m; Pp[t] = Pnow
            for sensor in range(2):
                avail = observed[t, :, sensor]
                residual = Y0[t, :, sensor] - par["bias"][sensor] - m[:, 0]
                variance = np.maximum(Pnow[:, 0, 0] + par["R"][sensor], 1e-6)
                ll += float(np.sum(-0.5*(np.log(2*np.pi*variance[avail]) +
                                             residual[avail]**2/variance[avail])))
                K = Pnow[:, :, 0]/variance[:, None]
                K *= avail[:, None]
                old_row = Pnow[:, 0, :].copy()
                m += K*residual[:, None]
                Pnow -= K[:, :, None]*old_row[:, None, :]
            mf[t] = m
            Pf[t] = (Pnow + Pnow.transpose(0, 2, 1))/2
        ms = mf.copy(); Ps = Pf.copy()
        cross = np.empty((T-1, n, 2, 2))
        for t in range(T-2, -1, -1):
            J = (Pf[t] @ A.T) @ inverse2(Pp[t+1])
            ms[t] += np.einsum("nij,nj->ni", J, ms[t+1]-mp[t+1])
            Ps[t] += J @ (Ps[t+1]-Pp[t+1]) @ J.transpose(0, 2, 1)
            cross[t] = Ps[t+1] @ J.transpose(0, 2, 1)
        W = ms[:, :, 0] - ms[:, :, 1]
        V = Ps[:, :, 0, 0] + Ps[:, :, 1, 1] - 2*Ps[:, :, 0, 1]
        C = cross[:, :, 0, 0] + cross[:, :, 1, 1] - cross[:, :, 0, 1] - cross[:, :, 1, 0]
        lag = W[:-1]
        response = W[1:] - float(fixed_b)*D[:-1]
        regressors = [lag, constant_reg, time_matrix]
        XtX = np.array([[np.sum(u*v) for v in regressors] for u in regressors])
        XtX[0, 0] += np.sum(V[:-1])
        Xty = np.array([np.sum(u*response) for u in regressors])
        Xty[0] += np.sum(C)
        coefficients = np.linalg.solve(XtX, Xty)
        new_rho = float(np.clip(coefficients[0], 0.01, 0.9995))
        par["rho"] = new_rho
        par["b"] = float(fixed_b)
        par["drift"] = coefficients[1:].copy()
        error = W[1:] - new_rho*lag - par["b"]*D[:-1] - (
            par["drift"][0] + par["drift"][1]*time_matrix)
        par["Q"] = max(float(np.mean(error**2 + V[1:] +
                                      new_rho**2*V[:-1] - 2*new_rho*C)), 0.1)
        for sensor in range(2):
            avail = observed[:, :, sensor]
            if sensor == 1:
                par["bias"][sensor] = float(np.mean((Y[:, :, sensor]-ms[:, :, 0])[avail]))
            residual = Y[:, :, sensor] - par["bias"][sensor] - ms[:, :, 0]
            par["R"][sensor] = max(float(np.mean(
                (residual**2 + Ps[:, :, 0, 0])[avail])), 0.1)
        par["initial_mean"] = ms[0].mean(axis=0)
        deviations = ms[0] - par["initial_mean"]
        par["initial_cov"] = Ps[0].mean(axis=0) + deviations.T @ deviations/n
        logs.append(dict(iteration=iteration+1, rho=par["rho"], b=par["b"],
                         Q=par["Q"], R0=par["R"][0], R1=par["R"][1],
                         bias1=par["bias"][1], nll=-ll/observed.sum()))
    print(drug, "seconds", round(time.perf_counter()-start, 1),
          "paired mean difference", round(bias1, 3),
          "paired difference SD", round(np.sqrt(pair_var), 3))
    print(pd.DataFrame(logs).iloc[[0, 3, 7, 15, len(logs)-1]].round(5).to_string(index=False))
    return dict(parameters=par, log=pd.DataFrame(logs), stay_ids=ids,
                times=times, smoothed=ms, covariance=Ps)

latent_fits = {}
latent_fits["insulin"] = fit_latent_record_model(
    panel, ("glucose_finger", "glucose_lab"), "insulin", -1.8)
latent_fits["norepinephrine"] = fit_latent_record_model(
    panel, ("map_arterial", "map_cuff"), "norepinephrine", 3.6)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 38: latent-state persistence diagnostic\n"
            "Fitted own two-sensor Gaussian state models, with stable random stay "
            "baselines, measurement noise, missing readings and known hourly drugs. "
            "Drug coefficients fixed to record-supported -1.8 and +3.6. "
            "This is a statistical diagnostic, not world-model training. Results:\n")
    for drug, result in latent_fits.items():
        f.write(drug + ": " + repr(result["parameters"]) + "\n")
```
