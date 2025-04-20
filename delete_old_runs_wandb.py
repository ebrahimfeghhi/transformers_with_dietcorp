import wandb

# Replace with your entity and project name
api = wandb.Api()
project = "Neural Decoder"
entity = "skaasyap-ucla"  # or your username

# Fetch runs
runs = api.runs(f"{entity}/{project}")

for run in runs:
    cer = run.summary.get("CER")  # adjust if your metric is named differently
    state = run.state  # "running", "finished", "crashed", etc.

    if cer is not None and cer > 0.4 and state != "running":
        print(f"Deleting: {run.name} | CER: {cer} | State: {state}")
        run.delete()
