The drop worked, so the model is back to the original. Two findings shape the next step:
- Drug atoms' dose and rate enter through columns 2 and 4 (zd, zr) of the shared atom numeric projection `embed.numeric.weight` (192×6). Measurement atoms put zeros there, and END atoms use only the bias. Only `encode_action` sees drug START atoms, so scaling those two columns changes only the drug token u and leaves every reading untouched.
- This amplifies the model's own learned dose→reading pathway, the one that already gives −2 for insulin ×2.

The risk is factual fit. Amplification also strengthens the response to factual dose variation, centred on the average rate rather than on the policy-expected rate. That could bias predictions at high or low glucose levels.

So I'll test empirically: scale both columns by A = 4, then run dose_edit and factual with model "both" on the same 100 tuning stays. If the insulin and NE shifts move toward the targets (−12.5, +6.25, +4.3, −2.2) and value NLL worsens by less than about 0.003, I'll tune A. If NLL worsens a lot, or the response barely moves, I'll drop this lever. First I save the edit and print the stay ids I need for the forward requests.

```python
with open('/work/notes.md','a') as f_: f_.write("- run283-284: z-token channel USELESS (prelude is pre-norm attention; dz-token->glucose sens ~0, NE ~0.01/unit). Dropped z edits.\n- Family1 test: scale embed.numeric.weight cols 2(zd),4(zr) (drug-only numerics) by A to amplify model's own dose pathway. Test A=4 with fwd dose_edit+factual both on sel1.\n")
Wn=np.asarray(load_original('embed.numeric.weight'),dtype=np.float64).copy(); A_amp=4.0
Wn[:,2]*=A_amp; Wn[:,4]*=A_amp
tp=type(load('embed.numeric.bias')); cv_=(lambda a: torch.as_tensor(a,dtype=torch.float32)) if tp is torch.Tensor else (lambda a: np.asarray(a,dtype=np.float32))
save_edit('embed.numeric.weight',cv_(Wn)); print('edited',edited(),'col norms',np.linalg.norm(Wn,axis=0).round(2))
print(','.join(str(int(s)) for s in sel1))
```