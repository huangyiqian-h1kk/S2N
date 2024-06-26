---
license: apache-2.0
base_model: hfl/chinese-roberta-wwm-ext-large
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: MLM_tuned_roberta_large
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# MLM_tuned_roberta_large

This model is a fine-tuned version of [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.7765
- Accuracy: 0.6219

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.36.0.dev0
- Pytorch 2.1.0
- Datasets 2.15.0
- Tokenizers 0.15.0
