The probe worked, and two findings change the plan.

**1. The activation is GELU.** act(0.5) = 0.3457 and act(−0.5) = −0.1543, which are exactly GELU's values. So any repurposed unit must stay deep in the positive linear regime or deep in the negative zero regime.

**2. u_k is a fixed lookup by rate, and an odd one.** For insulin-only bundles, u_k is identical for every bundle with the same rate (sd 0). It is not monotonic in rate:
- rate 1 gives a very different vector, probably its own rate-bin embedding, since the bin edges repeat;
- rates 0.5, 2, 3, 4 and 6 give nearly the same vector.

So the dose signal reaching the core is weak and irregular, which fits the model ignoring dose surprises.

There is also a small puzzle. u copies are nonzero (up to 0.34) at some bundles flagged as non-drug. These are probably END-only bundles, where an infusion record ends with no new record starting. Since u_k never varies within a rate, END atoms apparently don't carry the rate.

To build a reliable rate readout, including doubled rates up to about 20 and halved rates down to 0.25 that never appear in records, I need a CPU replica of `encode_action` checked against these exact u_k coordinates. First I need to know the bundle composition: which atom combinations occur, and whether u_k depends on n_atoms or END atoms.

```python
U = Z4[:, ucols]
nz = np.abs(U).sum(1) > 1e-6
print('nondrug bundles with nonzero u:', (nod & nz).sum(), 'of', nod.sum(), '; their n_atoms:', np.unique(na4[nod & nz], return_counts=True))
# do these nonzero-u nondrug bundles sit at drug end times (h+0.0833) ?
fr = np.round(hr4 % 1, 4)
print('frac part of hour for nondrug&nz:', np.unique(fr[nod & nz], return_counts=True))
# within insulin-only: does u depend on n_atoms at same rate?
for rv in [0.5, 1, 2]:
    for k in [1,2,3]:
        mm = m_io & np.isclose(ri, rv) & (na4==k)
        if mm.sum(): print('rate',rv,'n_atoms',k,'n',mm.sum(),'u[:4]',U[mm][0,:4].round(3),'sd',U[mm].std(0).max().round(4))
# distinct u vectors among nondrug&nz
uu = np.unique(np.round(U[nod & nz],4), axis=0); print('distinct u among nondrug-nz:', len(uu))
```