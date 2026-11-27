## Step 37: inspect the states file

**What the last result showed.** There are 5,218 bundle rows for 100 stays. Each row is 992 numbers: out (192), the 4 h slots (4 × 192) and z (32), plus the stay, hour, drug flags and atom count.

**What I will do next.**
- Load the file and look at its structure: bundles per stay, the hours, and the atom counts of drug bundles. Do END atoms appear, so that a drug bundle holds start plus END atoms?
- Look at the state magnitudes (norms of out and h).
- Pair bundles with the records.

That tells me how bundles are built, which I must reproduce in any local implementation.

**What would count as evidence.** If drug bundles show n_atoms = 2 when an infusion continues (a new start plus the previous one's END), END atoms are part of the action set.

```python
S = np.load('/work/fwd_3/states_original.npz')
print({k: S[k].shape for k in S.files})
X = S['x']; sid = S['stay_id']; hr = S['hour']; hi = S['has_insulin']; hn = S['has_norepinephrine']; na = S['n_atoms']
df_s = pd.DataFrame({'stay_id':sid,'hour':hr,'hi':hi,'hn':hn,'na':na})
s0 = df_s[df_s.stay_id==2020000064]
print(s0.head(40).to_string())
print("norms out:", np.linalg.norm(X[:,:192],axis=1)[:5], " h slots:", [np.linalg.norm(X[:,192+192*s:384+192*s],axis=1)[:3] for s in range(4)], " z:", np.linalg.norm(X[:,960:],axis=1)[:5])
print(tu[tu.stay_id==2020000064].head(40).to_string())
```