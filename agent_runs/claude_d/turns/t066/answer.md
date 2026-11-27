The architecture shows how the readout works. The edit pathway is action atom → embedding → `encode_action` → u_k → z posterior and prelude token → state. The readout is the value head's predicted mean (softmax over 17 bins times bin centres), averaged over 20 paths. A useful check before designing an edit is whether the model's weak response is already present at the first hour after an edited dose, or builds up only through the rollout feedback. A shortfall at the first hour would mean the dose-to-state pathway itself is too weak. I'll compare the model's per-stay-hour shifts from forward pass 1 with the per-hour true shifts already computed in the kernel. First I'll check the file layouts, keeping the output short.

```python
import glob
print(sorted(os.listdir('/work/fwd_1')), sorted(os.listdir('/work/fwd_2')))
de1=pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(de1.shape); print(de1.columns.tolist()); print(de1.head(3).to_string())
print(type(true_shift_ins), getattr(true_shift_ins,'shape',None))
if isinstance(true_shift_ins,(pd.DataFrame,pd.Series)): print(true_shift_ins.head(3))
```