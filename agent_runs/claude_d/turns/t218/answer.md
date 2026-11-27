This settles where the drift comes from:
- **Original model.** At hour 13 its rollout prediction equals its teacher-forced prediction (mean difference +0.2, the spread being z-sampling noise).
- **Edited model.** Its rollout prediction is 14 mg/dL higher on the same readings (median +16).
- **Same state.** No reading is drawn in hour 13 before that first glucose reading, so the state and head inputs are the same in both modes.

The edited head therefore gives a different answer for the same input depending on how it is run. The likeliest cause is numerical precision. E2's gate units sit around 86 for an active gate, and the dose signal is the small difference of two GELUs of large numbers, which is then multiplied back up. If the rollout runs in half precision (fp16 or bf16), that difference is mostly rounding error. The original model has no such cancellation and is unaffected.

The test is to run E2's head offline on the stored train-stay head inputs, with fp32 and with emulated reduced precision:
- If bf16 or fp16 gives a large positive bias on insulin-present glucose rows while fp32 does not, and the original head is unaffected, precision is confirmed. Any new edit must then keep all activations at ordinary magnitudes and avoid cancellation.

```python
import torch
for fn_ in ['head_probs','mean_from_A','nll_of','eval_edit','wc_from','headv','exp_tilt2']:
    try: print(fn_, inspect.signature(globals()[fn_]))
    except Exception as ex: print(fn_, 'ERR', ex)
print(type(HI), list(HI.keys()) if isinstance(HI, dict) else '')
print({k_: getattr(v_, 'shape', None) for k_, v_ in HI.items()} if isinstance(HI, dict) else '')
def head_mean(WC, W1, b1, W2, b2, trans, cen, dt=torch.float32):
    x = torch.tensor(WC).to(dt); W1t = torch.tensor(W1).to(dt); b1t = torch.tensor(b1).to(dt)
    W2t = torch.tensor(W2).to(dt); b2t = torch.tensor(b2).to(dt)
    hid = torch.nn.functional.gelu((x @ W1t.T + b1t).float()).to(dt)
    lg = (hid @ W2t.T + b2t).float() + torch.tensor(trans).float()
    p_ = torch.softmax(lg, -1).numpy()
    return (p_ * cen).sum(-1)
W1o, b1o = load_original('attr_heads.value_bin.0.weight').numpy(), load_original('attr_heads.value_bin.0.bias').numpy()
W2o, b2o = load_original('attr_heads.value_bin.2.weight').numpy(), load_original('attr_heads.value_bin.2.bias').numpy()
W1e, b1e = load('attr_heads.value_bin.0.weight').numpy(), load('attr_heads.value_bin.0.bias').numpy()
W2e, b2e = load('attr_heads.value_bin.2.weight').numpy(), load('attr_heads.value_bin.2.bias').numpy()
print(W1o.shape, W2o.shape, np.abs(W1e-W1o).max(), np.abs(W2e-W2o).max())
```