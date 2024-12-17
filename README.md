# warp-implementation

This repository implements **WARP (Weight Averaged Rewarded Policies)**, a novel alignment strategy for large language models (LLMs) introduced in the paper *"WARP: On the Benefits of Weight Averaged Rewarded Policies"* by Alexandre Ramé et al. (Google DeepMind). WARP optimizes the trade-off between KL regularization and reward maximization in Reinforcement Learning from Human Feedback (RLHF), improving alignment and performance of LLMs.

## Features
- **Dynamic Anchor with EMA**: Uses the exponential moving average of the policy as a dynamic anchor for KL regularization.
- **Spherical Interpolation (SLERP)**: Merges independently fine-tuned policies into a new enhanced model.
- **Linear Interpolation Towards Initialization (LITI)**: Balances KL and reward by interpolating between the merged model and the initialization.


## Structure
- config.yaml - Configuration file
- warp_training.py - Training script for the WARP model
- reward_training.py - Training script for the reward model
- prepare_data.py -  Data preparation script
- evaluate.py -  Validation script
- exp_results.ipynb - Experiment with hyperparameters

## Getting started
```
1. Clone the repository
First, clone the repository to your local machine:
git clone https://github.com/your-account/your-repository.git
cd warp_implementation

2. Install dependencies
Create and activate a virtual environment (it is recommended to use venv or conda):

# For venv
python -m venv env
source env/bin/activate  # For Windows, use `env\Scripts\activate`

# For conda
conda create --name myenv python=3.8
conda activate myenv
Install the required dependencies from the `requirements.txt` file:

pip install -r requirements.txt

3. Run training
Run the main training script:
python reward_training.py
python warp_training.py

4. Check results
After training is complete, you can find the saved model in the `results/warp_model` directory. Update the `config.yaml` file with the model path.
python evaluate.py
