GPT-2 SFT with RapidFireAI

Description:
Fine-tunes GPT-2 using Supervised Fine-Tuning (SFT) with LoRA adapters on a customer support chatbot dataset (Bitext). A reduced subset was used for Colab: 80 training examples and 20 evaluation examples. RapidFireAI was used to explore multiple hyperparameter and LoRA configurations, optimizing both token-level accuracy (eval loss) and sequence-level output quality (RogueL).

Dataset

Source: Bitext Customer Support Chatbot Dataset

Training: 80 examples (shuffled)

Evaluation: 20 examples (shuffled)

Model & Experiment

Base: GPT-2 (causal LM)

LoRA applied to target modules: c_attn and c_proj

Hyperparameters varied: rank (r), lora_alpha, lora_dropout, learning rate, and scheduler type

Training checkpoints evaluated with smoothed metrics for optimal early stopping

Results

Best Configuration (Config 1) at step 64:

Eval loss: 1.152

RogueL: 0.05553

Peak RogueL (step 52): 0.06156

Train loss (step 65): 1.356

Step 64 selected because it balances token-level accuracy and sequence-level output quality, avoiding overfitting observed at later steps

Usage

Open the Colab notebook
 to run the experiments

Modify peft_params, training_args, or dataset paths to run new configurations

Metrics plots are viewable via TensorBoard in the notebook
