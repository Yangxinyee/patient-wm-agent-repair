Run 72 rules out the state route for the clean dose channel. The direction written by the linear dose input is buried in large natural variation (natural sd about 97 against 0.7 per unit z). The dose shift exists only in the state just after a drug bundle and is wiped out at the next reading bundle. Later lags then propagate only through the readings the model generates itself. No direction in the state gives a selective, high-gain carrier.

I also see a constraint that applies to any fix. Factual doses vary about as much as the edited ones, so any edit that makes the model respond strongly to dose also changes its teacher-forced predictions. In the records, most dose variation follows the glucose tier, and the model has already absorbed that into its glucose dynamics. Adding the true drug effect on top would count it twice. Before searching for more levers, I need to know how large a dose response the 0.005-nat budget allows.

Next I will estimate that from data: the residuals of the original model on its factual predictions, against the recorded dose history. I will check:
- whether residuals follow the dose level or only the within-tier innovation;
- what Gaussian-approximate NLL cost comes from adding a fraction λ of the true one-step dose effect, for different centring points.

First I need to find which saved pass holds the original model's factual output.

```python
for k in [1,2,3,4]:
    d=f'/work/fwd_{k}'
    sm=json.load(open(d+'/summary.json'))
    print(k, sm.get('type'), sm.get('model'), sm.get('n_stays'), os.listdir(d))
    if sm.get('type')=='factual':
        print(json.dumps(sm['results'])[:1200])
```