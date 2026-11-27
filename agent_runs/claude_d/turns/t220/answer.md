The stored head inputs are complete: with_code at the prior-mean z, the transition logits, and bin centres for value bins 1–16, with class 0 meaning "no value". The reading table has insulin and norepinephrine at lags 1–3 for each reading.

I will now run the precision test, computing the predicted mean from bins 1–16 both raw and renormalised:
- I check which form reproduces the GPU's original predictions.
- I then compare edited and original heads in fp32, bf16 and fp16 on glucose_finger rows, with and without insulin in the previous hour, and on map_arterial rows with norepinephrine.

A jump of around +10 to +15 mg/dL under bf16 or fp16 on insulin rows, with fp32 near 0, would confirm that E2's large-activation cancellation is the cause.

```python
def head_mean2(WC, W1, b1, W2, b2, trans, cen, dt=torch.float32):
    x = torch.tensor(WC).to(dt)
    h1 = x @ torch.tensor(W1).to(dt).T + torch.tensor(b1).to(dt)
    hid = torch.nn.functional.gelu(h1)
    lg = (hid @ torch.tensor(W2).to(dt).T + torch.tensor(b2).to(dt)).float() + torch.tensor(trans).float()
    p_ = torch.softmax(lg, -1).numpy().astype(np.float64)
    raw = (p_[:,1:]*cen).sum(-1); ren = raw / p_[:,1:].sum(-1)
    return raw, ren, p_[:,0]
r32o, n32o, p0o = head_mean2(WC0, W1o, b1o, W2o, b2o, HI['trans'], HI['cen'])
print('match pred0: raw rms %.2f, renorm rms %.2f, mean p0 %.4f' % (np.sqrt(np.nanmean((r32o-rd_.pred0)**2)), np.sqrt(np.nanmean((n32o-rd_.pred0)**2)), p0o.mean()))
gfr = (rd_.event=='glucose_finger').values; mar = (rd_.event=='map_arterial').values
insp = (rd_.ins_1>0).values; nep = (rd_.ne_1>0).values
for dt_, nm_ in [(torch.float32,'fp32'),(torch.bfloat16,'bf16'),(torch.float16,'fp16')]:
    _, no_, _ = head_mean2(WC0, W1o, b1o, W2o, b2o, HI['trans'], HI['cen'], dt_)
    _, ne_, _ = head_mean2(WC0, W1e, b1e, W2e, b2e, HI['trans'], HI['cen'], dt_)
    b_ = ne_ - n32o; bo_ = no_ - n32o
    print(nm_, 'edit-orig32: gf ins %.2f | gf noins %.2f | map ne %.2f | map none %.2f ; orig(dt)-orig32 gf %.2f map %.2f' % (
        b_[gfr&insp].mean(), b_[gfr&~insp].mean(), b_[mar&nep].mean(), b_[mar&~nep].mean(), bo_[gfr].mean(), bo_[mar].mean()))
```