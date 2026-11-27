This changes the picture.
- **The old ridge decoders don't transfer.** Their response to the actual dose edits is only 0.12–0.19 of the dose change, not the 0.86 slope they showed on the perturbation samples. They are also very sensitive to z draws (sd 5.9 for insulin, 1.6 for NE).
- **No head-level decoder is affordable.** The head input moves too little with dose. The best minimum-variance decoder needs 5 u/h of noise for insulin and 1 mcg/kg/min for NE to get unit response. A correction read from the head input would wreck the factual fit.

Since the original state barely carries the dose at the head, I need a cleaner route for dose information to reach the value predictions. ARCHITECTURE.md mentions a "pending" input p (pending_mlp, filter_gate) used by the action heads. If p carries the currently running infusion rates, it is a direct dose signal at prediction time. I'll read the rest of the heads section to see what p contains and whether any weight-only route lets the value head use it.

```python
for i in range(58, 92): print(i, lines[i][:260])
```