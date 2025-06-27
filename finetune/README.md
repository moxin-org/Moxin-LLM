## Post-Training with Tülu 3

The open-source Tülu 3 dataset and framework are adopted for the model post-training. For our post-training, with our base model, we follow Tülu 3 to perform supervised finetuning (SFT) and then Direct Preference Optimization (DPO).

Specifically, we use the Tülu 3 SFT Mixture dataset from Tülu 3  to train our base model with  the SFT training method for two epochs and obtain our SFT model, following the default training configuration of the Tülu 3 8B SFT model. 
+ Dataset:  [Tülu 3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
+ Framework: [open-instruct](https://github.com/allenai/open-instruct)
+ Configuration: [Llama-3.1-Tulu-3-8B-SFT](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)

Refer to 'scripts/finetune/instruct_finetune/sft_finetune.sh' for more details. 

Next, we continue to train our SFT  model  on the Tülu 3 8B Preference Mixture dataset from Tülu 3  with the DPO training method to obtain our DPO model, following the same training configuration of the Tülu 3 8B DPO model. 
+ Dataset:  [Tülu 3 8B Preference Mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture)
+ Framework: [open-instruct](https://github.com/allenai/open-instruct)
+ Configuration: [Llama-3.1-Tulu-3-8B-DPO](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)

Refer to 'instruct_finetune/dpo_finetune.sh' for more details. 

##  Reinforcement Learning with GRPO

To enhance the CoT capabilities of our model, we adopt RL techniques similar to DeepSeek R1. We first use high quality reasoning data to SFT our instruct model.  The reasoning data mainly includes Openthoughts   and OpenR1-Math-220k. Next, we  adopt  the RL techniques in DeepSeek R1, i.e., GRPO to  finetune our model with RL.  We adopt the [DeepScaleR](https://github.com/agentica-project/rllm)  as our RL training framework.

We first use high quality reasoning data to SFT our instruct (DPO) model.
+ Dataset:  [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and  [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
+ Framework: [open-instruct](https://github.com/allenai/open-instruct)
+ Configuration: [Llama-3.1-Tulu-3-8B-SFT](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)

Refer to 'scripts/finetune/instruct_finetune/sft_finetune.sh' for more details. 

Next, we  adopt GRPO to  finetune our model with RL.
+ Framework, configuration and Dataset: [DeepScaleR](https://github.com/agentica-project/rllm)

Refer to 'reason_finetune/train_7b.sh' for more details. 
