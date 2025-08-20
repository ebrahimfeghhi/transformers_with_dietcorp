from llama_finetuning_hyperparam_search import train_model  # Replace with actual filename (no .py)
import os
import wandb

# Directory to save metrics and models
base_model_save_path = "/opt/dlami/nvme/saved_lora"
base_tokenizer_save_path = "/opt/dlami/nvme/saved_lora"
base_metrics_save_path = "/home/ubuntu/data/metrics"

# Define the r values to sweep over
r = 16
# Fixed settings for all other hyperparameters
learning_rate = [2e-4]
num_train_epochs = 2
batch_size = 16
gradient_accumulation_steps = 4

wandb_project = "llama_hyperparam sweep"

# Run sweep
for lr in learning_rate:
    
    lora_alpha = r  # Tie lora_alpha to r
    run_name = f"learning_rate_{lr}"
    model_save_path = os.path.join(base_model_save_path, run_name)
    tokenizer_save_path = os.path.join(base_tokenizer_save_path, run_name)
    metrics_save_path = os.path.join(base_metrics_save_path, f"{run_name}.txt")

    print(f"\n=== Starting run {run_name} ===\n")
    
    wandb.init(
        project=wandb_project,
        entity=None,
        name=run_name,
        config={
            "r": r,
            "lora_alpha": lora_alpha,
            "learning_rate": lr,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps
        },
    )

    metrics = train_model(
        metrics_save_path=metrics_save_path,
        model_save_path=model_save_path,
        tokenizer_save_path=tokenizer_save_path,
        r=r,
        lora_alpha=lora_alpha,
        learning_rate=lr,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    # Log metrics to wandb
    if isinstance(metrics, dict):
        wandb.log(metrics)
    else:
        wandb.log({"metric": metrics})

    wandb.finish()

    print(f"\n=== Finished run {run_name} ===")
    print(f"Metrics: {metrics}\n")
