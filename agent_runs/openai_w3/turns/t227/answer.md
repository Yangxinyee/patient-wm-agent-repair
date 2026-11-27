The restricted decoder slightly reduces factual state perturbations, but setting the action input to zero produces a large insulin carrier signal. That may be harmless only if attention reliably blocks it. I’ll test the signal after attention; appreciable leakage would need to be fixed before trusting the rollout.

```python
print("=== TRANSPORTED ZERO-ACTION SIGNAL ===")
_zero_u204 = np.zeros_like(U154)
_zero_z204 = posterior153(C154, E154, _zero_u204, post187)
_zero_tokens204 = tokens192.clone()
_zero_tokens204[:,5,:] = 0
_zero_tokens204[:,6,:] = F.linear(_zero_z204, tensor153(zin154), tensor153(zb154))
_zero_base204 = prelude155(_zero_tokens204, base192)
_zero_pilot204 = prelude155(_zero_tokens204, checkpoint193)
_zero_restricted204 = prelude155(_zero_tokens204, restricted_candidate203)
for _label204,_parts204,_dec204 in [
    ("pilot", _zero_pilot204, decoder192),
    ("restricted", _zero_restricted204, restricted199)]:
    _att204 = _parts204[3][:,3,:,:].detach().numpy()
    _vals204 = _parts204[2].detach().numpy() @ _dec204
    _transport204 = np.einsum("nsk,nkd->nsd", _att204, _vals204)
    print(_label204,
          "latent attention mean/max", _att204[:,:,6].mean(), _att204[:,:,6].max(),
          "transport mean/RMS", _transport204.mean((0,1)),
          np.sqrt(np.mean(_transport204**2, axis=(0,1))),
          "E RMS versus no-payload",
          float(torch.sqrt(torch.mean((_parts204[0]-_zero_base204[0])**2))))
print("\n=== FACTUAL TRANSPORT BY ACTION ===")
print("Y154 range:", Y154.min(0), Y154.max(0))
print("rates ranges:", rates187.min(0), rates187.max(0))
for _j204 in range(2):
    _present204 = rates187[:,_j204] > 0
    print("drug", _j204, "present count", int(_present204.sum()))
    for _maskname204,_mask204 in [("present",_present204),("absent",~_present204)]:
        if _mask204.any():
            print(_maskname204, "full/restricted signal RMS",
                  np.sqrt(np.mean(signal192[_mask204]**2,axis=0)),
                  np.sqrt(np.mean(_sig203[_mask204]**2,axis=0)))
print("carrier vectors orthogonality to original z_in/constant:",
      np.linalg.norm(zin_original154.T @ restricted199, axis=0),
      const199 @ restricted199)
print("candidate decoder norms:", np.linalg.norm(restricted199,axis=0))
```
