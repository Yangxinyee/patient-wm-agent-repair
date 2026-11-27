The recurrent replay matches the saved state closely. It also explains the tiny derivatives: the raw slots are large, so normalization attenuates small injections. I’ll use larger central differences over all 192 directions, then test the resulting correction. Consistent sensor responses across gains would confirm a usable calibration.

```python
import numpy as np, torch, time
torch.set_num_threads(8)
t186 = time.time()
n186 = len(i169)
eps186 = 8.0
J186 = np.zeros((4, 192), dtype=np.float64)

def response186(injections):
    # injections: number of directions x 192; each is tested on all retained states.
    injections = tensor153(injections)
    m = len(injections)
    tok = tensor153(tokens_orig169).repeat(m, 1, 1)
    inj = injections[:, None, :].repeat_interleave(n186, dim=0)
    Ep = preludeE185(tok, inj)
    hp, op = state_update82(
        hf169.repeat(m, 1, 1), Ep,
        mem169.repeat(m, 1, 1), mask169.repeat(m, 1),
        age_emb169.repeat(m, 1, 1), 0)
    vp = view171(hp, op, dt169.repeat(m))
    return arr183(phys171(vp)).reshape(m, n186, 4).mean(axis=1)

for start in range(0, 192, 16):
    axes = np.eye(192, dtype=np.float32)[start:start+16]
    plus = response186(eps186 * axes)
    minus = response186(-eps186 * axes)
    J186[:, start:start+len(axes)] = ((plus-minus)/(2*eps186)).T

u186, s186, vt186 = np.linalg.svd(J186, full_matrices=False)
print("Jacobian singular values", s186)
print("Jacobian row norms", np.linalg.norm(J186, axis=1))
ridge186 = (0.025*s186[0])**2
directions186 = J186.T @ np.linalg.solve(
    J186 @ J186.T + ridge186*np.eye(4), T171)
print("Direction norms", np.linalg.norm(directions186, axis=0))
print("Linear reconstructed target\n", J186 @ directions186,
      "\nTarget\n", T171)

base186 = response186(np.zeros((1,192)))[0]
for drug in [0,1]:
    gains = np.array([-2.0,-1.0,-0.5,0.5,1.0,2.0])
    shifts = response186(gains[:,None]*directions186[:,drug][None,:]) - base186
    print("drug",drug,"gains",gains,"\nsensor shifts\n",np.round(shifts,4))

u_factual186 = action_vectors87((1.0,1.0))
print("Factual action replay shape", tuple(u_factual186.shape),
      "RMS versus U154",
      float(np.sqrt(np.mean((arr183(u_factual186)-U154)**2))))
print("Cache versus original differences",
      [(n, float(np.max(np.abs(arr183(W[n])-arr183(load_original(n))))))
       for n in zero169 if n in W])
print("elapsed seconds",round(time.time()-t186,2))

with open("/work/notes.md","a") as f:
    f.write("\nPython runs 185-186; forwards 10. Recurrent replay RMS 0.002185; large raw-state norms explain tiny small-epsilon derivatives. Run 185 stopped only at a NumPy/Torch diagnostic mismatch, without edits. Run 186 computes an unrestricted 192-direction central-difference calibration at epsilon 8 and checks nonlinear sensor responses; all weight edits remain the factual-neutral zero-payload checkpoint.\n")
```
