# Ascend NPU Environment

This note records the local Ascend 910 setup used for phase 1 experiments.

## Conda Environment

The usable user-level environment is:

```bash
/home/zlong/anaconda3/envs/clt-npu-py39/bin/python
```

It was created under the current user's `~/anaconda3` with Python 3.9, because the conda base environment is Python 3.13 and the locally downloadable `torch-npu` wheels tested here were cp39 aarch64 wheels.

Installed core packages:

```text
torch             2.7.1
torch_npu         2.7.1.post2
numpy             1.26.4
PyYAML            6.0.3
scipy             1.13.1
importlib_metadata 8.7.1
```

`numpy<2` is intentional: the local CANN Python components still use NumPy APIs removed in NumPy 2.x.

## CANN Layout

Observed default toolkit:

```text
/usr/local/Ascend/cann -> /usr/local/Ascend/cann-8.5.1
/usr/local/Ascend/ascend-toolkit/latest -> /usr/local/Ascend/cann
```

The Ascend PyTorch version table in the official repository maps `torch-npu==2.7.1.post2` to PyTorch 2.7.1 and CANN 8.5.0: https://github.com/Ascend/pytorch

On this machine, however, pure `cann-8.5.1` is not sufficient for PyTorch-NPU because `torch_npu/_C*.so` links `libhccl.so`, while `libhccl.so` is absent from `/usr/local/Ascend/cann-8.5.1`.

Read-only root inspection found `libhccl.so` only in the older toolkit trees:

```text
/usr/local/Ascend/ascend-toolkit/8.2.RC1
/usr/local/Ascend/ascend-toolkit/8.3.RC1
/usr/local/Ascend/ascend-toolkit/8.3.RC1.alpha003
```

The current supported project wrapper therefore remains:

```bash
scripts/with_ascend_env.sh
```

which points to `/usr/local/Ascend/ascend-toolkit/8.2.RC1`.

Do not use the mixed workaround `cann-8.5.1 + 8.2 libhccl.so` for training. It can import `torch_npu`, but fails during NPU computation with TBE/TVM configuration incompatibilities.

## Verified Commands

NPU smoke:

```bash
ASCEND_RT_VISIBLE_DEVICES=5 \
PYTHONNOUSERSITE=1 \
scripts/with_ascend_env.sh \
/home/zlong/anaconda3/envs/clt-npu-py39/bin/python \
scripts/npu_smoke.py --device npu:0 --size 128
```

Short phase 1b run:

```bash
ASCEND_RT_VISIBLE_DEVICES=5 \
PYTHONNOUSERSITE=1 \
PYTHON=/home/zlong/anaconda3/envs/clt-npu-py39/bin/python \
DEVICE=npu:0 \
DATA_DIR=data/phase1b_easy_conda_npu \
OUTPUT_DIR=outputs/phase1b_direct_curve_conda_npu_smoke \
STEPS_LIST='100' \
EVAL_EXAMPLES=50 \
scripts/with_ascend_env.sh \
scripts/run_phase1b_direct_curve.sh
```

Verified short-run result:

```text
steps  dev    id_test  ood_test  loss   sec
100    0.580  0.500    0.460     2.797  0.9
```
