# Protocol details

## Algebra versions and splits

Historical experiments used `num_elements`; the fixed upstream exposes `num_operate`.
Fresh generation records the latter and is not a byte-identical reconstruction of the
old 90k/3k files. Use frozen cleaned splits for historical comparisons. Raw validation
overlap is rejected. Frozen export preserves the question sets and metadata but applies
a deterministic shuffle; retain its seed and output hashes when comparing models.

Fresh training is balanced across 30 templates and budgets 2–5; categories are randomly
selected within each budget. Validation cycles through 2–17 per template. Warmup selects
budget-2 training targets. Original warmup quotas were not uniform across templates;
frozen export preserves those original counts instead of imposing the fresh quotas.

Upstream contains 31 templates; the benchmark cohort excludes `array_cyclic_shift_Zn.yaml`.
The adapter passes the tree-rotation template's existing inverse map to its sampler in
memory, fixing a missing argument without editing the snapshot or its algebraic rules.
Python, NumPy and Python hash seeds are fixed because some upstream samplers iterate sets.

Inductive targets stay paired with the deductive inputs. Every example is distinct within
its prompt, absent from every target split, and from the same template/category with
budget/difficulty 2–3. Historical targets require historical examples. The independent
interpreter verifies supported example answers. Pools expand when needed to cover inverse
syntax. Ten examples do not guarantee unique identification of a hidden rule.

The independent interpreter supports 28 templates. D4 and Rubik use ordered reference
matching, not independent semantic proof. Final-answer scoring does not verify every CoT
step. Instance deduplication does not establish absence of pretraining contamination.

## SynLogic

Source: [MiniMax-AI/SynLogic](https://github.com/MiniMax-AI/SynLogic), commit
`d8c527fd17edb739172619efb9b681805fc74b8d`; paper:
[SynLogic](https://arxiv.org/abs/2505.19641).

The fresh-12 adapter retains the experimental task parameter ranges and English prompt
fixes. Training and validation draw independent instances from the same distributions.
This is not the algebra difficulty-OOD split or the full official SynLogic benchmark.

The patch chooses English prompt branches, fixes a spatial-tree English description and
allows operation-generation timeouts to propagate. The builder checks commit and patch
state. Per-attempt seeds, bounded arithmetic, deadlines, duplicate checks and rejection
counters constrain generation. Missing dependencies fail immediately. Capacity exhaustion
fails the build instead of filling it with duplicates. A completed manifest is required
before treating a dataset as ready; interrupted output directories may be incomplete.

Sudoku and cryptarithm judges accept constraint-valid alternative answers. Model-generated
code is never executed. Verification metadata is stored outside model-visible prompts.
Generated references pass both upstream and adapter checks. Changing workers does not
change task seed assignment; changing split sizes can change subsequent rejection histories.

The final answer uses one closed, non-nested `<answer>...</answer>` block at the end
of the response. Its contents are passed directly to the task judge. Nested or multiple
answer blocks, missing closing tags and trailing prose are rejected. When no answer
tags occur, one terminal boxed expression or fenced block is also accepted. Scoring
requires a completed generation and does not search reasoning for a correct answer.

## Training and logging

The launcher targets public VERL v0.5.0. This is a portable implementation, not a claim
that every historical runtime used that commit. The trace patch transfers vLLM termination
metadata to the reward/archive boundary, without replacing PPO or GRPO. It is scoped to
AlgeScope's patched naive reward manager and is not intended for unrelated reward workloads.

All four model labels use explicit System/User/Assistant text with the tokenizer's BOS
when defined. Training and standalone evaluation share this rendering. Native instruct
chat templates are not silently substituted. Base and RFT checkpoints are distinct
experimental conditions and must be reported separately.

With one PPO epoch, 128 prompts and an 8-prompt PPO minibatch imply 16 optimizer minibatches
per rollout step. GPU changes preserve the global batch, sample count and sampling/data
settings. KL is disabled; algebra advantage standard-deviation normalization is disabled,
while SynLogic enables it. These are explicit choices, not a universal GRPO definition.

OLMo/Gemma runtime upgrades are outside this release.
A compatible transformer/vLLM implementation and GPU smoke test are still required before
full training. No private framework, cluster scheduler or machine-specific configuration
is included.

The summary requires the expected question set and equal sample counts. It distinguishes
average@k from pass@k. Algebra ID is 2–5 and OOD is 6–17, with the source difficulty field
reported. SynLogic retains task-specific difficulty. A single top-k-limited sample is not
pass@k. Variable-length token arrays are encoded as JSON strings in VERL reward logs to
avoid accidental numeric aggregation.

## Sources and validation scope

- [Algebrarium](https://github.com/Linzwcs/Algebrarium): algebraic systems, templates and generators.
- [SynLogic](https://github.com/MiniMax-AI/SynLogic): external generators and verifiers; MIT license.
- [VERL v0.5.0](https://github.com/volcengine/verl/tree/v0.5.0): training and rollout framework; Apache-2.0 license.

The independent algebra checks, SynLogic adapter and English patch were extracted from
the experiment implementations. Portable IO, launch/evaluation interfaces and the VERL
trace bridge were organized for this repository. Upstream attribution remains applicable.

CPU checks cover data construction, answer contracts, termination metadata handling,
aggregation and configuration composition. This repository organization does not itself
constitute a GPU training run or a regeneration of the complete historical datasets.

The preparation checks exercised 30 algebra templates with 30 warmup, 240 training and
480 validation targets, followed by paired inductive conversion and example-isolation
checks. SynLogic checks covered all 12 task families with one question per split/task.
A separate capacity check generated 3,000 training and 100 validation questions for
each of the 30 algebra templates. Configuration composition covered all four model
labels and all three benchmarks against the pinned public VERL schema. No GPU training
or model evaluation was launched as part of this code organization.
