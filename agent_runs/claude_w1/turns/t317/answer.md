The inspection shows the saved edit (EDIT v1) is badly broken in the tool. The edited factual NLL is 18.3 in fwd_5 and 16.3 in fwd_6, against 1.519 for the original model. The local replica predicted a cost of only about +0.03.

My leading explanation is the edit's design. It parks repurposed hidden units at very large biases (about 199) and cancels them with large logit offsets (about 119). The tool's head input must differ from my replica's input, probably through the sampled latent z, so the cancellation fails.

Next, a cheap check:
- per-event NLL change and predicted-mean change between edited and original in fwd_5 and fwd_6;
- how the head input is built (replica source, plus the architecture notes on the head).

If the mean shifts are huge on some codes, the gating or cancellation is the culprit, and I'll redesign with moderate magnitudes. The new design would pair two units whose gate terms cancel. If instead the shifts are small, the fault is elsewhere, such as indexing.

```python
import inspect
for k in [5, 6]:
    e = pd.read_csv(f'/work/fwd_{k}/factual_edited.csv'); o = pd.read_csv(f'/work/fwd_{k}/factual_original.csv')
    assert (e[['stay_id','hour','event']].values == o[['stay_id','hour','event']].values).all()
    m = o[['event']].copy(); m['dmu'] = e.predicted_mean - o.predicted_mean; m['dnll'] = e.nll - o.nll
    g = m.groupby('event').agg(n=('dnll','size'), dnll=('dnll','mean'), dnll_med=('dnll','median'),
                               dmu=('dmu','mean'), dmu_sd=('dmu','std'), dmu_q99=('dmu', lambda s: s.abs().quantile(.99)))
    print('fwd', k); print(g.round(3).to_string())
fn = [k for k, v in globals().items() if callable(v) and not k.startswith('_') and getattr(v, '__module__', None) == '__main__']
print('funcs:', ' '.join(fn)[:1500])
if 'head_mean' in globals():
    src = inspect.getsource(head_mean).splitlines()
    print('\n'.join(l[:150] for l in src[:45]))
arch = open('/model/ARCHITECTURE.md').read().splitlines()
hits = [i for i, l in enumerate(arch) if any(s in l.lower() for s in ['attr_head', 'head input', 'query', 'value_bin', 'embed_transition', 'z_head', 'prior'])]
for i in hits[:30]:
    print('A%d:' % i, arch[i][:170])
```