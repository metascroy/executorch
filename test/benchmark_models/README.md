# Reproducing ExecuTorch errors on torchbench models

1. Install torchbench models in a conda environment that is already set-up with executorch.  Installation instructions for torchbench can be found
[here](https://github.com/pytorch/benchmark), but the relevant bit is shown below (note that python refers to python3).

```bash
git clone https://github.com/pytorch/benchmark
cd benchmark
python install.py --continue_on_fail
```

2. Make sure the executorch_delegation_runner.py, report_generation.py, and run_torchbench_delegations.py files are in the same directory as the benchmark repro you cloned in step 1.  The directory structure should be something like this:
```
some_directory
|-- executorch_delegation_runner.py
|-- report_generation.py
|-- run_torchbench_delegations.py
|-- benchmark
|   |-- ...
```

3. Run the following command.  The `--write_dir` flag indicates where outputs files will be placed, and the `--exist_ok` flag means it is OK if this directory already exists.  `--delegations` is a list of delegations you want to lower torchbench models to.  At least one delegation must be specified.  Available options are: no_backend xnnpack xnnpack_quantized mps coreml lite.
```
python run_torchbench_delegations.py --write_dir ./runs --exist_ok --delegations no_backend xnnpack xnnpack_quantized mps coreml lite
```
