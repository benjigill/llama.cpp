---
name: build-check
description: Build the fork and run the CPU test suite in a Haiku subagent, returning only pass/fail and the first errors. Use after a merge, cherry-pick or code change, or when asked to build, compile or run the tests locally.
argument-hint: "[targets or test regex, or 'cuda']"
context: fork
agent: build-runner
model: haiku
---

Build and test this repo, then report in your standard format.

Caller arguments (may be empty): $ARGUMENTS

- Empty: default CPU configure, full build and `ctest -L main`.
- `cuda`: CUDA build in `build` with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA_QUANTS=q8_0-q4_0`, if `nvcc` exists. Then run `build/bin/test-backend-ops -b CUDA0 -j 8 -o MUL_MAT,GATED_DELTA_NET,SSM_CONV,FLASH_ATTN_EXT` and report any line that is not OK.
- Names of targets (e.g. `llama-server test-backend-ops`): build only those.
- Anything that looks like a test name or regex: build, then `ctest -R '<regex>'`.
