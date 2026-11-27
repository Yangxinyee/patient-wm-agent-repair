## Step 21: norepinephrine starts when MAP falls below 65

**What the last result showed.**
- Using the latest MAP reading, with arterial preferred when both fall in the same hour, norepinephrine starts with certainty when MAP < 64, about half the time at 64–65, and only 1–12% above 65.
- The half-way value at 64–65 again suggests a threshold at 65 on the unrounded reading.
- Some starts above 65 remain unexplained. Perhaps the rule uses the minimum of both readings, an older reading, or random starts.

**What I will do next.**
1. Check whether "start if the minimum of the latest arterial and cuff MAP is below 65" explains the starts above 65.
2. Look at titration while running: the hour-to-hour rate change against the latest MAP. Is there a target band? Is the step a multiplicative or additive noise component? Is that noise i.i.d.?
3. Look at stopping rules.

**What would count as evidence.** A titration step that is a deterministic function of MAP plus i.i.d. noise would give me the same surprise-instrument design as for insulin.

```python
st = xn[(xn.ne_prev==0)].copy()
st['mn'] = np.fmin(st.last_map_arterial.where(st.age_map_arterial==st[['age_map_arterial','age_map_cuff']].min(1)), st.last_map_cuff.where(st.age_map_cuff==st[['age_map_arterial','age_map_cuff']].min(1)))
sub = st[st.mn.notna()]
print("START by min-of-latest:", sub.groupby(pd.cut(sub.mn,[0,60,63,64,65,66,67,68,70,75,200])).ne_rate.apply(lambda s:(s>0).mean()).round(3).to_dict())
# starts above 65 under art-pref: what's age of reading, which readings
a = st[(st.lm_art_pref>66) & (st.ne_rate>0)]
print("n starts with art-pref>66:", len(a), " age art:", a.age_map_arterial.value_counts().head(5).to_dict(), " age cuff:", a.age_map_cuff.value_counts().head(5).to_dict())
print(a[['h','last_map_arterial','age_map_arterial','last_map_cuff','age_map_cuff','ne_rate']].head(15).to_string())
# start rate distribution
print("start rate dist:", st[st.ne_rate>0].ne_rate.describe().round(3).to_dict())
print(st[st.ne_rate>0].ne_rate.value_counts().head(10).to_dict())
```