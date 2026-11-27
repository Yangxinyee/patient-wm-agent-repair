This repeats the register audit, which still favors column 6; it contains no recurrent-response results. I’ll run the bounded replay check and latent-response calibration next, reusing results if that calculation already completed. Consistent replay and drug-specific responses would support a candidate edit; unstable responses would reject this pathway.

```python
import time
import itertools
import numpy as np
import torch

torch.set_num_threads(4)
torch.set_grad_enabled(False)

if "replay_scores161" not in globals():
    e_cal161 = torch.as_tensor(ecache152[(False, 2)], dtype=torch.float32)
    u_cal161 = torch.as_tensor(ucache152[(True, 2, 3)], dtype=torch.float32)
    state_cal161 = torch.as_tensor(S, dtype=torch.float32)
    rates_cal161 = np.asarray(rates152)
    indices_all161 = np.arange(len(S))

    def replay_inputs161(i, include_current=False, age_scale=1.0):
        sid = int(stays156[i])
        history = np.flatnonzero((stays156 == sid) & (indices_all161 < i))
        if not len(history):
            raise ValueError("Cannot replay initial bundle")
        previous = int(history[-1])
        mem_indices = history[-128:]
        if include_current:
            mem_indices = np.concatenate([mem_indices, [i]])[-128:]
        h = state_cal161[previous, 192:960].reshape(1, 4, 192)
        dt = torch.tensor([(hours156[i]-hours156[previous])*3600],
                          dtype=torch.float32)
        memory = e_cal161[mem_indices].unsqueeze(0)
        ages = torch.tensor((hours156[i]-hours156[mem_indices])*age_scale,
                            dtype=torch.float32).unsqueeze(0)
        return (h, dt, e_cal161[i:i+1], u_cal161[i:i+1],
                state_cal161[i:i+1, 960:992], memory, ages)

    def call_core161(args, age_mode):
        first, second = core_step136(*args, age_mode=age_mode)
        return (first, second) if first.ndim == 3 else (second, first)

    audit_indices161 = []
    for drug_index161 in (0, 1):
        pool161 = np.flatnonzero(
            (rates_cal161[:, drug_index161] > 0)
            & (hours156 >= 6) & (hours156 <= 18))
        audit_indices161.extend(
            pool161[np.linspace(0, len(pool161)-1, 3, dtype=int)])
    audit_indices161 = np.unique(audit_indices161)

    t161 = time.perf_counter()
    replay_scores161 = []
    for current161, age_scale161, mode161 in itertools.product(
            (False, True), (1.0, 60.0, 3600.0), ("key", "input")):
        he161, oe161, hs161 = [], [], []
        for i161 in audit_indices161:
            hh161, oo161 = call_core161(
                replay_inputs161(int(i161), current161, age_scale161), mode161)
            gold161 = state_cal161[i161, 192:960].reshape(1, 4, 192)
            he161.append(float(torch.mean((hh161-gold161)**2)))
            oe161.append(float(torch.mean((oo161-state_cal161[i161, :192])**2)))
            hs161.append(float(torch.mean(gold161**2)))
        replay_scores161.append((
            float(np.sqrt(np.mean(he161))),
            float(np.sqrt(np.mean(oe161))),
            float(np.sqrt(np.mean(he161)/np.mean(hs161))),
            current161, age_scale161, mode161))
    replay_scores161.sort()
    core_config161 = replay_scores161[0][3:]

print("=== REPLAY: h RMS, out RMS, relative h RMS, current memory, age scale, mode ===")
for row161 in replay_scores161[:8]:
    print(row161)

if "jacobian161" not in globals() and replay_scores161[0][2] < 0.20:
    probe_indices161 = np.unique(np.concatenate([
        audit_indices161, np.asarray(probe153[:12], dtype=int)]))
    perturb161 = torch.cat([
        torch.zeros(1, 32), torch.eye(32)*0.05, -torch.eye(32)*0.05])
    jacobian161 = []
    t_jac161 = time.perf_counter()
    for i161 in probe_indices161:
        args161 = replay_inputs161(
            int(i161), core_config161[0], core_config161[1])
        batch_args161 = []
        for k161, value161 in enumerate(args161):
            expanded161 = value161.expand((65,) + tuple(value161.shape[1:]))
            batch_args161.append(
                expanded161 + perturb161 if k161 == 4 else expanded161)
        hh161, oo161 = call_core161(batch_args161, core_config161[2])
        responses161 = []
        for event161 in ("glucose_finger", "map_arterial"):
            pred161 = readout153(
                hh161, oo161, int(stays156[i161]), float(hours156[i161]), event161, 1)
            responses161.append(
                ((pred161[1:33]-pred161[33:65])/0.10).numpy())
        jacobian161.append(np.stack(responses161))
    jacobian161 = np.stack(jacobian161)
    A161 = jacobian161.reshape(-1, 32).astype(np.float64)
    gram161 = A161.T @ A161
    ridge161 = 0.05*np.trace(gram161)/32
    latent_directions161 = {}
    for drug161, target161 in (
            ("insulin", (-1.988349, 0.0)),
            ("norepinephrine", (0.0, 3.288830))):
        v161 = np.linalg.solve(
            gram161 + ridge161*np.eye(32),
            A161.T @ np.tile(target161, len(jacobian161)))
        latent_directions161[drug161] = v161.astype(np.float32)
    print("Finite-difference seconds:", round(time.perf_counter()-t_jac161, 2))

if "latent_directions161" in globals():
    print("\n=== RESPONSE-BASED DIRECTIONS ===")
    print("Jacobian shape:", jacobian161.shape)
    print("Leading singular values:", np.linalg.svd(
        jacobian161.reshape(-1, 32), compute_uv=False)[:8])
    for drug161, v161 in latent_directions161.items():
        achieved161 = jacobian161 @ v161
        print(drug161, "norm", float(np.linalg.norm(v161)),
              "mean glucose / MAP", achieved161.mean(axis=0),
              "10–90% responses", np.quantile(achieved161, [0.1, 0.9], axis=0))
else:
    print("Replay insufficiently accurate; no directions fitted.")

print("ACTIVE EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Recurrent-response calibration after repeated Python 160 output\n"
            "- Retained previous-value column 6; alternatives worsened both readouts.\n"
            "- Audited memory inclusion and age conventions against exposed recurrent states.\n"
            "- Finite-difference latent directions, if available, are diagnostic only.\n"
            "- No weights changed; future-lag and GPU validation are still required.\n")
```