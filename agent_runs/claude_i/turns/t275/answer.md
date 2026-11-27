The attractor's factual cost is acceptable at insulin bundles: glucose moves by a median 0.18 mg/dL and MAP by 0.30. At norepinephrine bundles it is larger, a median 1.8 mg/dL on glucose, probably because head 3's usual z value is removed. A matching mean value could restore that.

The real problem is leverage. A unit change in head 3's value moves the next glucose prediction by only 0.11 mg/dL (insulin bundles) to 0.26 mg/dL (norepinephrine bundles). MAP moves even less (0.03 to 0.08), and the glucose and MAP directions overlap (cosine 0.5 to 0.7). With the value change bounded by LayerNorm, that is a fraction of a mg/dL, far short of the several mg/dL needed.

The question is whether the bottleneck is the value channel or the state-to-prediction map itself. I'll test the second directly. I'll add a common vector to all four slot outputs of the prelude, by temporarily shifting the prelude FFN output bias in WT, at 20 insulin bundles. Then I measure the gradient of the next glucose and MAP predictions, its consistency, and the response along the best direction at norms 1 to 30. A roughly linear response of several mg/dL at moderate norm means amplifying the head-3 value along that direction would work. A saturating, sub-mg/dL response means the state's grip on the predictions is the bottleneck, and the head must be edited too.

```python
bkey = P+'ffn.2.bias'; b_orig = WT[bkey].clone()
def pred_with_bias(k, d):
    WT[bkey] = b_orig + d
    try: return next_pred(k, ztok_add=zero)
    finally: WT[bkey] = b_orig
selX = ki[::60][:20]; JX = np.zeros((len(selX),192,2)); B0x = np.zeros((len(selX),2))
for a_,k in enumerate(selX):
    B0x[a_] = pred_with_bias(k, torch.zeros(192))
    for i in range(192): JX[a_,i] = pred_with_bias(k, I192[i]) - B0x[a_]
gX = JX[:,:,0].mean(0); cX = (JX[:,:,0]@gX)/(np.linalg.norm(JX[:,:,0],axis=1)*np.linalg.norm(gX))
mX = JX[:,:,1].mean(0)
print('X2 sens gf: per-bundle |g| med %.2f, mean |g| %.2f, cos med %.2f; MAP mean |g| %.2f; cos(gf,MAP) %.2f'%(np.median(np.linalg.norm(JX[:,:,0],axis=1)), np.linalg.norm(gX), np.median(cX), np.linalg.norm(mX), gX@mX/np.linalg.norm(gX)/np.linalg.norm(mX)))
gproj = gX - (gX@mX)/(mX@mX)*mX; dX = torch.tensor(-gproj/np.linalg.norm(gproj), dtype=torch.float32)
for s_ in [1,3,10,30]:
    d = np.array([pred_with_bias(k, s_*dX) - B0x[a_] for a_,k in enumerate(selX[:10])]); print(s_, d.mean(0).round(2), d[:,0].std().round(2))
```