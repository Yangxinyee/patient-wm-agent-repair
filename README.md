# Can AI Agents Repair Patient World Models?

Code, result files and the agents' full transcripts for the paper *Can AI Agents Repair Patient World Models?* by Xinye
Yang, Yuli Wang and Harrison Bai (preprint: link to follow). Corresponding author: Harrison Bai,
<harrison.bai@cuanschutz.edu>. Repository: <https://github.com/Yangxinyee/patient-wm-agent-repair>.

AI agents that edit a patient world model's weights, an instance of weight-space programming, can repair its drug
effect on the question asked, yet leave it wrong where no one asked. Patient world models fit to health records misread
what drugs do, because doses follow the patient's condition. The paper gives two AI agents, Claude and Codex, a world
model trained on a simulated intensive care unit where the true drug effects are known, and asks them to edit its
weights, without gradients or the truth, to correct its predicted effects of doubling and halving insulin and
norepinephrine. A known-truth test scores every edit on the questions
asked, on doses and measurements no one asked about, and on every output head. In our runs:

- On doubled insulin, 2 of 23 submissions come within a tenth of the true effect by 95% interval, both
  Claude's on the main world's seed-0 model (0.96 and 0.98 of the truth). On that model the unedited
  model reads 0.13 and our reference script 0.90.
- Even these fail where no one asked: every correction, the reference script included, fails when insulin is stopped,
  and 19 of the 20 insulin edits fall short when it is tripled.
- On unchecked outputs, 6 of the 17 submissions on other models pass the fit check yet raise
  total log loss beyond the spread between training seeds; Claude's first edit raises it by 6.3 nats per stay.
- Replaying the agents' estimation code shows that 3 of the 4 seed-0 runs not within the band had
  the right estimate and an edit that fell short of it.

Clinical world-model corrections should be scored with known truth, where the corrector was not asked to look, and so
should any claim that an agent repaired a network by writing its weights.

## What is here and what is not

```
agent/            the agent harness, its sandboxed Python kernel, the forward-pass tool and what the agents were given
agent_runs/       every agent run the paper counts, in full, and the replays of the agents' estimation code (see below)
logs/             which instruction each run in the further worlds received (more_worlds/assignment.json)
paper/            numbers.tex, the generated table rows, Figures 1-5 as printed, and the scripts that redo them
results/          the result files every number, table and figure of the paper is computed from
scripts/          evaluation, the estimator audit, the information-matched scripts, the paper's numbers, and the check
```

`MANIFEST.tsv` lists every file with its size and SHA-256.

**The simulator and the world model will be released separately.** That release, *the simulator code* below, covers the
ICU simulator and its truth replay, the event-stream world model with its expected-dose pathway, its training and
fine-tuning, the rollout evaluation, the trained checkpoints and the synthetic records. Everything here reaches that code through one module,
`agent/backend.py`, configured by `world_model.json` (copy `world_model.example.json`). All data are synthetic.

**What runs from this repository alone, and what does not.** Step 1 below (the paper's numbers, table rows and figures
from the released result files) runs here and is checked byte for byte. Every result file was computed by the scripts of
steps 2-4 and by the agents' runs; those steps simulate, roll out or evaluate a model and cannot be rerun without the
simulator code. Their outputs are released as computed. Four steps call the simulator code's internals and are released
with it, not here: installing an estimate in the expected-dose pathway (in the main and the further worlds), the
fine-tuning control, and building the further worlds.

Not in this package: the agents' input records and model files (written by `agent/prepare.py` from the simulator
code's release), the models the agents submitted (edited copies of the world model's weights), and the arrays, tables and
weights the agents wrote in their work folders.

## Setup

- Analyses: Python 3.13 with `numpy` 2.5, `polars` 1.44, `torch` 2.13, `safetensors` 0.8, `scikit-learn` 1.9 and
  `scipy` 1.18 (`requirements.txt`), plus the simulator code's requirements for the steps that need it.
- Figures: Python 3.12 with `matplotlib` 3.11, `reportlab` 5.1 and `numpy` (`requirements-figures.txt`). The figures use
  Arial metrics; with Arial missing, Liberation Sans is used.
- `world_model.json` gives where the simulator code lives and which Python runs it, its environment variables, and the
  module behind each neutral name (`sim`, `WorldModel`, `rollout`, ...; listed in `agent/backend.py`).
- The two worlds of the main study are `e03` (drugs act) and `n03` (no-effect world). Seed 0's unedited model is the one
  every correction starts from.

## Reproducing the paper

### 1. Numbers, tables and figures from the released results (CPU, about a minute)

Run from the repository root, with `requirements-figures.txt` installed (the numbers need only `numpy`):

```bash
python scripts/paper_numbers.py            # every number in the text and tables -> paper/numbers.tex, paper/generated/
python paper/make_figures.py               # Figures 2-5 -> paper/figures/case2_*.pdf
python paper/make_teaser.py                # Figure 1 -> paper/figures/fig1_teaser.pdf
python scripts/check_outputs.py            # all three, then each output against the shipped copy (MANIFEST.tsv)
```

`check_outputs.py` overwrites the shipped outputs with the redone ones and compares bytes; the only fields it blanks
first are the times a PDF records (`/CreationDate`, `/ModDate`) and the document ID Figure 1's PDF makes from them. `paper/make_teaser.py` also renders PNG previews with Ghostscript (`gs`).

Every result in the paper's text and tables is a macro of `paper/numbers.tex`, rounded half-up from the value as
computed; the only numbers typed in the paper's LaTeX source are design constants (simulator parameters, sizes, seeds,
budgets and hyperparameters) and values quoted from other work. The tables are typeset in the paper's LaTeX source from
these macros.

| Paper | Content | Result files | Analysis (writes the result files) |
|---|---|---|---|
| Figure 1 | The test and the paper's three results at a glance | `paper/numbers.tex`, `agent_eval.jsonl` | `paper_numbers.py`, `agent_eval.py` |
| Figure 2 | Insulin x2 readout over the truth for every correction of the seed-0 model | `readout.jsonl`, `agent_eval.jsonl` | `readout.py`, `scripted_report.py`, `agent_eval.py` |
| Figure 3 | Predicted and true change against the dose factor | `generalize.json`, `readout.jsonl`, `agent_eval.jsonl` | `generalize.py` + `generalize_summary.py`, as Figure 2 |
| Figure 4 | Change in held-out loss on every output head | `model_parts.jsonl` | `model_parts.py` |
| Figure 5 | Insulin x2 readout over the truth, by hour, and each edit's own no-effect shift | `readout.jsonl`, `agent_eval.jsonl`, `generalize.json` | as Figures 2 and 3 |
| Table 1 | Notation | -- | -- |
| Table 2 | Every correction and submission on insulin: doubled, tripled, stopped, total NLL | `submissions.json`, `readout.jsonl`, `generalize.json`, `model_parts.jsonl` | `submissions.py` (from `agent_eval.jsonl`, `generalize.json`, `model_parts.jsonl`, `told_runs_summary.json`, `more_worlds_summary.json`) |
| Table 3 | Estimate and installation on the same stays | `decompose_same_stays.json` | `decompose_same_stays.py` |
| Table 4 | Each main-world run's transcript: estimate, edit, caveat | `agent_runs/*/final_answer.md`, `agent_runs/*/tool_calls.jsonl` | the harness (`agent/harness.py`) |
| Table 5 | Final answers coded against the held-out verdicts | `transcript_codes.json` | coded by hand from the final answers |
| Table 6 | The agent runs: turns, Python runs, forward passes, cost, declined replies | `agent_runs/*/state.json` | the harness (`agent/harness.py`, `agent/budget.py`) |
| Table 7 | The three main-world runs per agent with intervals | `agent_eval.jsonl`, `generalize.json`, `model_parts.jsonl` | `agent_eval.py`, `generalize.py` + `generalize_summary.py`, `model_parts.py` |
| Table 8 | Every correction of the seed-0 model: asked, unasked, total NLL, no-effect shift | `readout.jsonl`, `agent_eval.jsonl`, `generalize.json`, `model_parts.jsonl`, `null_seed_batch.json` | as Tables 2 and 17 |
| Table 9 | Every submission with verdicts, the paired script difference and declined replies | `submissions.json` | `submissions.py` |
| Table 10 | Asked contrasts over three training seeds | `readout.jsonl`, `agent_eval.jsonl`, `ft_value_nll.json` | `readout.py`, `agent_eval.py`, `ft_value_nll.py` |
| Table 11 | Interventions no agent was asked about | `generalize.json` | `generalize.py`, `generalize_summary.py` |
| Table 12 | How far the training records' insulin doses reach | `dose_support.json` | `dose_support.py` |
| Table 13 | 95% intervals of the unasked readouts | `generalize.json` | `generalize_summary.py` |
| Table 14 | Tripled insulin by dose bin | `x3_split.json` | `x3_split.py` |
| Table 15 | Each estimate's implied shift at x2 and x3 | `x3_identification_ci.json` | `x3_identification_ci.py` |
| Table 16 | Change in held-out loss per head | `agent_parts.jsonl`, `model_parts.jsonl` | `agent_parts.py`, `model_parts.py` |
| Table 17 | Implied insulin x2 shift of each estimator in 21 inert worlds | `null_seed_batch.json` | `null_seed_batch.py` |
| Appendix text | Told runs and further worlds; same-shape references; replay along effect strength and in other worlds | `told_runs_summary.json`, `more_worlds_summary.json`, `same_shape/`, `appendix_checks_readouts.json`, `graded_summary.json`, `world_install/`, `budget_sweep.json`, `strength_sweep.json` | `told_runs_summary.py`, `more_worlds_summary.py`, `same_shape.py`, `appendix_checks_readouts.py`, `graded_summary.py`, `budget_sweep.py`, `strength_sweep.py`; `world_install/` by the install step (simulator code) |

Figures 2-5 are drawn by `paper/make_figures.py` and Figure 1 by `paper/make_teaser.py`, both from these files and
`paper/numbers.tex`.

Analysis scripts are in `scripts/`; result files in `results/`.

### 2. Re-running the analyses (needs the simulator code)

| Script | What it computes | Compute |
|---|---|---|
| `scripts/agent_eval.py <run dir>` | Held-out readout of an agent's submission (the four dose contrasts, 600 stays) and its value NLL → `results/agent_eval.jsonl` | GPU |
| `scripts/agent_parts.py <run dirs>` | Every output head's held-out loss, unedited against edited → `results/agent_parts.jsonl` | GPU |
| `scripts/model_parts.py <world> <label> <run> <loader>` | The same for any correction, with the total NLL per stay → `results/model_parts.jsonl` | GPU |
| `scripts/generalize.py <world> <run> <loader> <tag> <seed> doses\|starts` | Rollouts of interventions no agent was asked about: other dose factors, the second instrument, other start hours | GPU |
| `scripts/generalize_summary.py` | Scores those rollouts against the simulator's truth → `results/generalize.json` | CPU |
| `scripts/per_stay.py` | Per-stay agreement with the truth, and a dose-only predictor → `results/per_stay.json` | CPU |
| `scripts/readout.py <world> <method> <seed>` | Readout of the control corrections' rollouts → `results/readout.jsonl` | CPU |
| `scripts/paired_readout.py` | Paired held-out comparison of two installed corrections → `results/paired_readout.json` | CPU |
| `scripts/probe.py extract\|fit_part\|fit` | Linear probe of the surprise dose in the unedited model's state → `results/probe.json` | CPU |
| `scripts/edits.py steer ...`, `scripts/stateedit.py`, `scripts/edited_rollout.py` | The steering controls (rank-one state edits), their factual NLL and their held-out rollout | CPU, GPU |
| `scripts/model_size.py` | Parameters and tensors of the unedited model → `results/model_size.json` | CPU |
| `scripts/dose_units.py` | The expected-dose pathway's dose unit per drug (the typical training rate) → `results/dose_units.json` | CPU |

### 3. The estimator audit

Each estimator returns a lag profile; Eq. (2) of the paper turns it into the insulin x2 shift it implies on the
readout's scale. In a no-effect world that shift is the estimator's bias; in a main world, minus the simulator's true
shift on the same stay-hours, it is its error.

| Script | What it does |
|---|---|
| `scripts/replay_estimator.py <run> <world> <last run>` | Replays an agent's own estimation code, unchanged, in a fresh sandbox on another world's records → `agent_runs/replay/` |
| `scripts/replay_bootstrap.py <run> <world> <first> <last>` | The same replay on copies of the training records resampled by stay → `results/replay_bootstrap/` |
| `scripts/scripted_bootstrap.py <world> <draws>` | The information-matched script's estimator on the same kind of copies |
| `scripts/paired_bootstrap.py`, `scripts/paired_summary.py` | All three estimators on the same copies, for paired differences → `results/paired_bootstrap*` |
| `scripts/fresh_null_seeds.py check\|run\|agents` | Five fresh no-effect worlds (simulator seeds 4101-4105, tuning 4201-4205), records formed as the original ones (`check` confirms the original world's records are reproduced value for value); the script's estimator (`run`) and the agents' replayed estimators (`agents`) on each → `results/fresh_null_seeds*.json` |
| `scripts/fresh_null_bootstrap.py <run> <first> <last>` | The agents' replayed estimators on stay-resampled copies of the fresh worlds → `results/fresh_null_bootstrap/` |
| `scripts/audit_eq2.py check\|truth\|replay\|script` | Eq. (2) on Claude's replayed lag profile, Codex's own Eq. (2) value and the script's kernel, in the no-effect worlds and in six main worlds (seeds 5101-5105 and the original), with the simulator's true shift → `results/audit_eq2/` |
| `scripts/no_effect_pooled.py` | Pooled bias and error over six worlds, with stratified and two-stage bootstrap intervals and the band verdicts → `results/no_effect_pooled.json` |

### 4. The information-matched scripts (no agent)

| Script | What it does |
|---|---|
| `scripts/scripted_estimate.py <world>` | A fixed two-stage least-squares estimator of each drug's effect by lag, instrumented by the dose's deviation from the mean dose in the same glucose or pressure cell → `results/scripted/estimate_<world>.json` (CPU) |
| `scripts/scripted_tune.py <variant>` | Sizes one gain per drug with the agents' own dose-edit tool on the 400 tuning stays, until the tuning readouts match the estimate → `results/scripted/tune_<variant>.json` (GPU). The step that installs the estimate in the expected-dose pathway is released with that pathway in the simulator code. |
| `scripts/scripted_report.py` | Collects the baselines → `results/scripted_baseline.json` |

## The agents

| Step | Command | What it does |
|---|---|---|
| Inputs | `python -m agent.prepare` | Writes what the agent sees, per world: the unedited model's weights (or only tensor names and shapes for the input-side arm), the vocabulary, `ARCHITECTURE.md`, and the 30,000 training and 400 tuning stays as CSV. It writes no held-out stay, truth, simulator code or world name. |
| Run | `python -m agent.harness init <dir> --agent claude\|openai --world e03\|n03 --arm white\|black`, then `python -m agent.harness run <dir> [--responder fake\|module:Class]` | One run. The agent's Python runs in a sandbox (`agent/kernel.py`: bubblewrap, no network, no GPU, memory and CPU caps; at most 300 runs). Forward passes go through `agent/forward.py` and `agent/fwd_job.py` on the GPU (at most 50). The run ends at a final answer. `--responder fake` (the default) is a scripted dry run (`agent/dryrun/fake_agent.py`). A real run plugs in a responder with the interface in `agent/responder.py`; the paper's runs passed each turn to the agents' command-line tools (`claude -p`, `codex exec`) with their own tools switched off and the system prompt in `agent/assets/agent_system_prompt.md`. |
| Cost | `python -m agent.budget <runs dir>` | API dollars per run, from the token counts. |

`agent/assets/` holds what the agents received, verbatim: the task message (`message.txt`), the told runs' message
(`message_unqueried.txt`, naming the interventions scored beyond the four asked), the rerun's message in further world 3
(`message_w3r.txt`), the two arms' API notes (`arm_white.txt`, `arm_black.txt`), the architecture note, the data README,
the notes the harness sent after a declined reply (`refused_note.txt`, `refused_note_w3r.txt`) and the system prompt
(`agent_system_prompt.md`).

### The runs (`agent_runs/`)

Claude's runs are `claude_*`, Codex's `openai_*`. Every run below had weight access unless marked.

| Folders (Claude; Codex) | World and model | Instruction | In the paper |
|---|---|---|---|
| `claude_a`, `claude_d`, `claude_e`; `openai_a`, `openai_d`, `openai_e` | main world, seed-0 model | main | runs 1-3 (Tables 6 and 7) |
| `claude_c`; `openai_c` | main world, seed-0 model, inputs only (input-side arm) | main | Table 6 |
| `claude_b`; `openai_b` | no-effect world | main | Table 6 |
| `claude_h`; `openai_h` | fresh no-effect world fA | main | Table 6 |
| `claude_g`; `openai_g` | fresh no-effect world fB | main | Table 6 |
| `void_claude_f`; `void_openai_f` | fresh no-effect world fA | main | first runs there, stopped unscored after a missing file |
| `claude_i`, `claude_k`; `openai_i`, `openai_k` | main world, seed-1 model | told | Tables 2 and 9 |
| `claude_j`, `claude_l`; `openai_j`, `openai_l` | main world, seed-2 model | told | Tables 2 and 9 |
| `claude_w1`, `claude_w2`, `claude_w4`, `claude_w5`; `openai_w1` to `openai_w5` | further main worlds 1-5 | main or told, alternating (`logs/more_worlds/assignment.json`) | Tables 2 and 9 |
| `claude_w3`; `claude_w3r` | further main world 3 | main; the rerun's (`message_w3r.txt`) adds the task's nature | paused after many declined replies, not counted; the rerun, reported apart |

Inside a run the worlds keep the names the agent saw: `e03` for every world with drug effects, `n03` for every no-effect
world. Each run folder holds `state.json` (counts and cost), `final_answer.md`, `turns/tNNN/` (the prompt the agent
received, its answer, and the turn's metadata), `tool_calls.jsonl` (every Python run with its code and output),
`tokens.jsonl`, `requests/` (every forward-pass request and its result) and `work/` (the text files under 2 MB the agent
wrote: its helper modules and notes). The harness's own logs are not included. The transcripts are as recorded, except
that identifiers of the computing environment (paths, user names, provider prefixes of model names) are replaced by
neutral placeholders such as `<repo>` or left out. `agent_runs/replay/` holds the outputs of
the agents' estimation code replayed unchanged on the main and the no-effect world's records.

## License

The code (`agent/`, `scripts/`, `paper/*.py`, `paper/figlib/`) is under the MIT License (`LICENSE`). The agents'
transcripts and the result files (`agent_runs/`, `results/`, `logs/`, and `paper/numbers.tex`, `paper/generated/`,
`paper/figures/`) are under the Creative Commons Attribution 4.0 International License (`LICENSE-data`).
