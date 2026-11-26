# AlgeScope

Our dataset construction is based on
[Algebrarium](https://github.com/Linzwcs/Algebrarium). We provide three protocols:
**deductive algebraic reasoning**, **inductive algebraic
reasoning**, and **SynLogic**, with a common VERL training and evaluation interface.

## Installation

Use Python 3.10+ on Linux and run commands from the repository root. Data preparation
only requires a CPU:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Prepare the GPU training environment separately as described below.
The optional upstream `Algebrarium/src/case-print.py` helper requires Python 3.12+.

## 1. Deductive reasoning

Questions supply an algebraic definition and explicit computation rules. The three
categories are expression evaluation, single-variable equations, and systems of group
equations. The 30 benchmark templates are listed in `configs/algebra_templates.json`.

```bash
python -m algescope.data.deductive --output data/deductive
```

| Split | Default size | Fresh-generation protocol |
|---|---:|---|
| Warmup | 9,000 | Budget-2 source questions selected from training |
| RL train | 90,000 | Generation budgets 2–5 |
| Validation | 3,000 | Generation budgets 2–17; excluded from training/warmup |

The defaults allocate 3,000 training, 100 validation and 300 warmup questions per
template.

Fresh generation does not reproduce the exact historical questions, quotas or ID/OOD
counts. To reuse an existing experiment's cleaned question sets and difficulty labels:

```bash
# SOURCE contains train.parquet, validation.parquet and warmup.parquet.
python -m algescope.data.deductive \
  --source-dir SOURCE --output data/deductive_frozen
```

This preserves questions, answers, prompts and metadata, with a deterministic output
shuffle. Validation overlap is an error; historical questions are not silently replaced.

## 2. Inductive reasoning

Inductive inputs retain the definition but replace explicit computation rules with ten
distinct question/answer examples from the same template and question category. Target
questions, answers, split membership and sample IDs remain paired with the deductive source.

```bash
python -m algescope.data.inductive \
  --deductive-dir data/deductive --output data/inductive
```

All warmup/train/validation targets are excluded from the example pool. Examples are
sampled without replacement. Targets using `inv(...)` receive an inverse-syntax example.
Examples contain final answers, not full reasoning traces.

For historical `num_elements` targets, provide historical examples explicitly:

```bash
python -m algescope.data.inductive \
  --deductive-dir data/deductive_frozen \
  --example-source HISTORICAL_EXAMPLES.parquet \
  --output data/inductive_frozen
```

The example source may contain individual examples or structured `inductive_cases`.
See [protocol details](docs/protocols.md) for verification and version boundaries.

## 3. SynLogic

We use [MiniMax-AI/SynLogic](https://github.com/MiniMax-AI/SynLogic), introduced in
[*SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning
and Beyond*](https://arxiv.org/abs/2505.19641). Our experiment regenerates a **12-task
subset**, rather than evaluating the complete
[official released dataset](https://huggingface.co/datasets/MiniMaxAI/SynLogic).

```bash
python -m pip install -r requirements-synlogic.txt
mkdir -p external
git clone https://github.com/MiniMax-AI/SynLogic.git external/SynLogic
git -C external/SynLogic checkout --detach d8c527fd17edb739172619efb9b681805fc74b8d
git -C external/SynLogic apply ../../patches/synlogic_english.patch

python -m algescope.data.synlogic \
  --source external/SynLogic --output data/synlogic --workers 4
```

The task families are operation, three Dyck-language variants, web of lies, object
properties, goods exchange, spatial reasoning, spatial tree reasoning, Sudoku,
cryptarithms and arrow mazes. Each contributes 750 warmup, 7,500 training and 250
validation questions: **9,000 / 90,000 / 3,000** in total.

Warmup uses easier parameters. Training and validation use the same parameter
distribution with independent instances. Canonical-instance and prompt deduplication
span all three splits. References must pass the upstream and bounded final-answer
verifiers. The answer contract is one closed, non-nested `<answer>...</answer>` block
at the end of the response; generation rejections are recorded.

## Common VERL training

One FSDP entry point serves **Qwen2.5-32B, Qwen3-30B-A3B, GLM-4-32B-Base and
Mistral-Small-24B-Base**. Supply the intended Base or completed RFT checkpoint explicitly;
model labels do not change the optimization settings. The newer OLMo/Gemma runtime
adaptations are not bundled in this release.

The public compatibility target is [VERL v0.5.0](https://github.com/volcengine/verl/tree/v0.5.0),
commit `8fdc4d3f202f41461f4de9f42a637228e342668b`. This is not a claim that all historical
runs used that exact commit. Prepare compatible CUDA/PyTorch/vLLM dependencies following
VERL's installation instructions, then install the checkout and the trace patch:

```bash
git clone --branch v0.5.0 https://github.com/volcengine/verl.git external/verl
git -C external/verl checkout --detach 8fdc4d3f202f41461f4de9f42a637228e342668b
git -C external/verl apply ../../patches/verl_v050_trace.patch
python -m pip install -e external/verl
python -m pip install tensorboard

# Print the complete command without launching training.
python -m algescope.training.launch \
  --model qwen25 --checkpoint CHECKPOINT \
  --benchmark deductive --data data/deductive \
  --output outputs/qwen25_deductive --gpus 8 --tp 4

# Add --execute after reviewing the command and preparing the GPU environment.
```

Other labels are `qwen3`, `glm4` and `mistral`. Use `--benchmark inductive` or
`--benchmark synlogic` with the corresponding data directory. For SynLogic, export
`SYNLOGIC_SOURCE` to the absolute path of the pinned, patched upstream checkout.
Multi-node runs use `--nodes N --gpus GPUS_PER_NODE` on an existing Ray cluster with
shared code/data paths. No platform resource allocation is performed by this repository.

| Setting | Common recipe |
|---|---|
| Training budget | 200 rollout steps |
| Prompt batch / responses per prompt | 128 / 8 |
| PPO minibatch / epochs per rollout | 8 prompts / 1 |
| Optimizer | AdamW, LR `1e-6`, betas `(0.9, 0.999)`, weight decay `0.01` |
| Policy loss | Vanilla PPO, clip `0.2`, token-mean |
| KL / entropy penalties | Disabled / 0 |
| Training sampling | Temperature 1, top-p 1, top-k disabled |
| Prompt / response limits | 8,192 / 8,192 tokens |
| Input handling | Full splits; overlength inputs fail rather than being filtered |
| GRPO advantage standard-deviation normalization | Off for algebra; on for SynLogic |
| Validation | Greedy, step 0 and every 20 steps |
| Checkpoints | Every 100 steps |

## Evaluation and records

RL, validation and standalone evaluation share the final-answer reward implementation:

```bash
python -m algescope.training.evaluate \
  --checkpoint HF_CHECKPOINT --data data/deductive/validation.parquet \
  --output outputs/evaluation --tp 4

python -m algescope.training.summarize \
  --data data/deductive/validation.parquet \
  --traces outputs/evaluation/traces.jsonl --output outputs/evaluation/scores.csv

# Aggregate all saved RL validation steps:
python -m algescope.training.summarize \
  --data data/deductive/validation.parquet \
  --traces outputs/qwen25_deductive/validation --output outputs/validation_curve.csv
```

Greedy evaluation is the default. Specify `--samples 8 --temperature 1.0` for sampled
evaluation. The summary separately reports average@k and pass@k, verifies complete
question coverage and equal sample counts, and groups by template and ID/OOD. The
source difficulty field is retained to avoid pooling incompatible generator versions.

Warmup collection uses the evaluator with `warmup.parquet`, `--samples 16` and
`--temperature 0.6`. Only complete, verified correct outputs are candidates for RFT.
This release supplies source-data and collection interfaces, not a new SFT trainer.

RL writes `validation/{step}.jsonl`, `rollouts/{step}.jsonl`, TensorBoard events,
resolved Hydra configuration and a console log. Traces include question IDs, full
output text, scores, template/difficulty, token IDs/counts and actual finish/stop reasons.
Variable-length token arrays in VERL logs are JSON-encoded strings; the standalone
evaluator stores JSON arrays. Export FSDP checkpoints with VERL's model-merging utility
before passing them to the standalone HF/vLLM evaluator.

## Layout and checks

```text
Algebrarium/           Unmodified fixed upstream source
algescope/data/        Deductive, paired inductive and SynLogic builders
algescope/rewards/     Shared final-answer verifiers
algescope/training/    VERL launcher, evaluation, trace handling and summaries
configs/              Template list and portable scientific parameters
patches/              Pinned public dependency adaptations
docs/                 Protocol details and verification scope
tests/                CPU checks
```

```bash
python -m unittest discover -s tests -v
```

Generated datasets, weights, checkpoints, trajectories, credentials and platform
configuration are not included. All AlgeScope-maintained documentation is in English.
