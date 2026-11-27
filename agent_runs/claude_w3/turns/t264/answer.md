Passes 5 and 6 were factual checks of an edited model: 3 stays, then 100 stays. Predictions differ from the original even at the first reading, so the edit (or an earlier version of it) changes the model's ordinary behaviour. I'll read the edited model's NLL from both summaries to see how much factual fit that cost.

```python
for k in [5, 6]:
    s = json.load(open(f'/work/fwd_{k}/summary.json'))
    r = s['results']
    for m in ['original', 'edited']:
        print(k, m, r[m]['value_nll_per_value'], {e: r[m]['by_event'][e]['nll'] for e in r[m]['by_event']})
    print(k, 'mtime', os.path.getmtime(f'/work/fwd_{k}/summary.json'))
print('edit files mtimes:', sorted(set(round(os.path.getmtime(x)) for x in glob.glob('/work/edited/*'))))
```