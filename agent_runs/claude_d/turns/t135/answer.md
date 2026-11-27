None of the view variants beat the documented formula, so the view is probably right as written. One remaining suspect is the memory layout of the 4 h slots inside the 992-vector. If they are stored dimension-major rather than slot-major, the per-dimension flow decay and the slot mean would be mixed up, which would give exactly this kind of moderate, global error. Below I test the alternative reshape and check the dtype of the saved states. If the error collapses to the level of z noise, that was the bug.

```python
print(S3['x'].dtype)
_S3x_backup = S3x
X_alt = S3x.copy(); X_alt[:,192:960] = S3x[:,192:960].reshape(-1,192,4).transpose(0,2,1).reshape(-1,768)
S3x = X_alt; score(predict(FR,typ=1,gap_scale=10.0),'h_dimmajor')
S3x = _S3x_backup; score(predict(FR,typ=1,gap_scale=10.0),'h_slotmajor')
```