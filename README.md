

# Single run
example
```bash
python src/main.py train.lr=0.01 train.batch_size=32
```


# Sweep
1. Create a sweep on Weights & Biases:
    ```bash
    wandb sweep src/sweep.yaml
    ```
2. Start the sweep agent:
    ```bash
    wandb agent --count <max_iter> <SWEEP_ID>
    ```


    Replace `<SWEEP_ID>` with the actual sweep ID returned from the previous command.
    Replace `<max_iter>` with the maximum number of iterations you want to run. (early_terminate in yaml did not work at least in bayes method)

