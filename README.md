# 601.471_FinalProj_Sp25

For proof of concept, we ran the experiment described in our paper under concept_proof.


To run full pipeline, run:
1. phi_01.sh: Original fine-tuned Phi-1.5 model, no corruption
2. pad_noslide_phi_01.sh: Applying padding to fine-tuned Phi-1.5 model without sliding window
3. pad_slide_phi_01.sh: pplying padding to fine-tuned Phi-1.5 model with sliding window
_*Note, these are all in llm-unlearn-eco_

All correlated results were placed in output.

Where our implementation is:
1. Sliding Window Classifier (eco/attack)
2. Evaluation (eco/scripts/evaluate_tofu.py)
3. Configuration changes (.yaml files)
4. Tofu Dataset (padding)
5. tofu.py
