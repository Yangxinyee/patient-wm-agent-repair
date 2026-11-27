The mediation check changes the picture. In the original model, adding z26 leaves the dose-innovation coefficient of the state change unchanged (out 0.035 → 0.036, h 8.03 → 8.02). So the dose reaches the state through the action token in the prelude, not through z.

In the edited model, z posterior sensitivity also rose only 1.4×, although z_post reads the encoder output u linearly. So amplifying the inputs ×5 changed u's dose sensitivity by only about 1.4×.

Two explanations fit:
- The bin tables carry most of the dose information, but not along my linear profile, so the 1-D ramp edit did nothing.
- The encoder saturates; the asymmetric increments (x2 smaller than x1/2) point this way for the z columns.

Next I amplify all deviations of bin rows 1–16 from their mean, in both the dose and rate tables, by ×3. I do this with 15 rank-1 edits per table along an orthonormal basis of the centred subspace, without the z-column edits. If the insulin response grows close to 3×, the bin tables carry the signal and the encoder is roughly linear. If it hardly moves, the encoder saturates and I must look elsewhere.

```python
clear_ops()
C16 = np.eye(16) - np.ones((16,16))/16
w_, V_ = np.linalg.eigh(C16)
basis = V_[:, w_ > 0.5]   # 16 x 15 orthonormal, sum-zero
print(basis.shape, np.abs(basis.sum(0)).max())
for tname in ['embed.bins.dose.weight', 'embed.bins.rate.weight']:
    for j in range(basis.shape[1]):
        vv = np.zeros(17); vv[1:] = basis[:, j]
        rank1(tname, vv, 2.0, side='out')
print(len(ops()), ops()[0], ops()[-1])
```