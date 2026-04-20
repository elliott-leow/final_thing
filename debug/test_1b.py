"""Local smoke test of the debug experiments on OLMo-2 1B Instruct.

Runs the three experiments with N_TRAIN=15 and N_TEST=6 (small for CPU).
Self-judge for all experiments (no independent judge on 1B).

Purpose: verify the pipeline runs end-to-end before the 7B Colab run.
Est runtime: ~30-45 min on CPU.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import (
    set_seeds, cleanup,
    run_experiment_1_truncation,
    run_experiment_2_independent_judge,
    run_experiment_3_generation_time_direction,
)

set_seeds(42)

# ── Config (reduced for 1B CPU) ──
MAIN_MODEL_ID = 'allenai/OLMo-2-0425-1B-Instruct'
# For the "independent" judge on 1B, we DON'T load a second model (RAM);
# we just skip Experiment 2 and note it requires 2+ models.
RUN_EXP_2 = False

N_TRAIN = 15  # direction extraction
N_TEST = 6    # steering evaluation
ALPHAS = [-5, 0, +5]

print(f'Loading {MAIN_MODEL_ID}...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MAIN_MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    attn_implementation='eager')
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

N_LAYERS = model.config.num_hidden_layers
LAYERS = sorted(set(list(range(0, N_LAYERS, 2)) + [N_LAYERS - 1]))
MID_LAYER = LAYERS[len(LAYERS) // 2]
print(f'{N_LAYERS} layers, sampling {LAYERS}, MID={MID_LAYER}', flush=True)

stim = json.load(open('v2_clinical_cold.json'))
stim_train = stim[:N_TRAIN]
stim_test = stim[N_TRAIN:N_TRAIN + N_TEST]

# Self-judge for exp 1 and 3
judge_model, judge_tokenizer = model, tokenizer

results = {}

# ── EXPERIMENT 1: truncation ablation ──
results['exp1'] = run_experiment_1_truncation(
    model, tokenizer, stim_train, stim_test, LAYERS, MID_LAYER,
    ALPHAS, judge_model, judge_tokenizer)
cleanup()

# ── EXPERIMENT 2: skip on 1B (needs second model) ──
if RUN_EXP_2:
    print('Skipping Experiment 2 on 1B (needs 2 models, low RAM)', flush=True)
else:
    print('\n(Skipping Experiment 2 — runs only on Colab with 2 models loaded)',
          flush=True)

# ── EXPERIMENT 3: generation-time direction ──
results['exp3'] = run_experiment_3_generation_time_direction(
    model, tokenizer, stim_train, stim_test, LAYERS, MID_LAYER,
    ALPHAS, judge_model, judge_tokenizer)
cleanup()

os.makedirs('debug', exist_ok=True)
json.dump(
    {k: (v if isinstance(v, dict) else v) for k, v in results.items()},
    open('debug/test_1b_results.json', 'w'),
    indent=2, default=str)
print('\n\nDONE. Results saved to debug/test_1b_results.json', flush=True)
