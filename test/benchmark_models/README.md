# Reproducing ExecuTorch errors on torchbench models

1. Install torchbench models in your conda environment that is already set-up with executorch.  Installation instructions for torchbench can be found
[here](https://github.com/pytorch/benchmark).  The relevant part of the installation instructions is shown below (note that python refers to python3).

```bash
git clone https://github.com/pytorch/benchmark
cd benchmark
python install.py --continue_on_fail
```

2. Make sure the executorch_delegation_runner.py file and the run_torchbench_delegations.py script are in the same directory as the benchmark repro you cloned in step 1.  The directory structure should be something like this:
```
some_directory
|-- executorch_delegation_runner.py
|-- run_torchbench_delegations.py
|-- benchmark
|   |-- ...
```

3. Set the appropriate `delegations` and `write_dir` in the `__name__ == "__main__"` section of the run_torchbench_delegations.py script.  The `write_dir` is where the script will output errors and pte files for each model in torchbench.  The `write_dir` MUST ALREADY EXIST prior to running the script.

4. Run the run_torchbench_delegations.py with `python run_torchbench_delegations.py`
