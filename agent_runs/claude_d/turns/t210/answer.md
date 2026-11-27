The hourly breakdown separates two problems.

**Insulin.**
- At hour 13 the edited model is about right: -3.9 against a target of -3.2.
- After that the shift plateaus near -8 and then falls back, while the target keeps building to about -17.
- The lag coefficients of the rollout show this. The edit gives about -1.5 at lag 1, -1.1 at lags 2 and 3, about -0.7 at lag 4 and then nothing. The records give -2.0, -2.4, -2.1, -1.7, -1.2, -0.9 and so on.
- So the direct one-hour channel works and is roughly linear: lag 1 is -1.52 for x2 and -1.48 for x1/2.
- What is missing is memory. The model's state forgets the dose after about two bundles, so effects at lag 3 and beyond can only come from the model carrying its own sampled readings forward. It does that only weakly and briefly.
- The x1/2 shortfall is the same problem: the later-hour response even drops below the original model's.

**Norepinephrine.**
- The x2 response is almost nothing (lag 1 about +0.2), while x1/2 responds strongly.
- This fits the dose signal in the state saturating at high rates. Recorded rates top out near 1.0, so a doubled rate may land in the top input bin.

**Options.**
- Scaling up the head channel can push the hourly means toward the targets. My rough estimate is that the mean shift itself costs little factual NLL; E2's excess cost probably comes from side effects.
- Giving the state a dose memory with the right decay would be more faithful. That requires knowing exactly how a dose event writes into the slow slot dimensions.

Before choosing, I will re-read the state-update part of ARCHITECTURE.md. The question is whether a dose value can be written additively into slot dimensions with a suitable decay time, without the recurrent block overwriting it.

```python
t_ = open('/model/ARCHITECTURE.md').read()
print(len(t_))
print(t_[:7000])
```