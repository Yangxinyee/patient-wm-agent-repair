The pooling variants barely change the error (0.099–0.117), so the encoder's form is probably right. The remaining suspect is the meaning of the dumped state.

The architecture says out = out_norm(mean over slots of the new h). If the dumped h is the state actually carried forward, that identity should hold exactly in the dump. If it fails, the dump stores some other version of h, which would explain a context error.

```python
H4 = X4d[:, 192:960].reshape(-1, 4, 192); O4 = X4d[:, :192]
pred_out = ln_(H4.mean(1), 'core.out_norm', P0)
print('out vs out_norm(mean h): rms', ((pred_out - O4)**2).mean().sqrt().item(), ' out rms', (O4**2).mean().sqrt().item())
print('corr', np.corrcoef(pred_out.numpy().ravel(), O4.numpy().ravel())[0,1])
```