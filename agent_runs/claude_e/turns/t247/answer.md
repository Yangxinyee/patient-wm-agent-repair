Run 227 appended to the notebook. It then failed at the comparison step, because `ORIG` holds torch tensors rather than numpy arrays, so nothing was saved and the edit is still `ins_only(-2.5)`. I will convert everything to numpy first. Then I'll confirm that `ins_only(-2.6)` changes only the value-head tensors and leaves the transition table at its original values, and save it. That makes the next factual checks run on −2.6.

```python
def to_np(x):
    return x.detach().cpu().numpy().astype(np.float64) if hasattr(x,'detach') else np.asarray(x,dtype=np.float64)
for k in ['W0','b0','W2','TR']:
    a_=to_np(E26[k]); o_=to_np(ORIG[k]); c_=cur[k]
    print(k,'max|E26-orig|',round(float(np.abs(a_-o_).max()),4),'max|E26-saved|',round(float(np.abs(a_-c_).max()),4))
for k in ['W0','b0','W2']:
    save_edit(KMAP[k], torch.tensor(to_np(E26[k]),dtype=torch.float32))
print(edited())
chk_=[float(np.abs(load(KMAP[k]).numpy().astype(np.float64)-to_np(E26[k])).max()) for k in ['W0','b0','W2']]
print('saved check',chk_)
```