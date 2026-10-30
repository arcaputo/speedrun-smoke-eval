# speedrun-smoke

A Kupanga QA eval for the modded-nanogpt speedrun: the candidate's train_gpt.py compiles and trains on one H100.
The agent's sandbox has no GPU, so a bug that only shows on one (an Inductor error while compiling, a CUDA
kernel, a shape mismatch in the compiled graph) would otherwise first appear in the 8×H100 benchmark, where a
crash can't be sent back to the agent. This one is: its critique carries the first traceback.

It runs the record folder the candidate changed exactly as speedrun-8xh100 does (bench.py here is a copy of
speedrun-8xh100's), with torchrun --nproc_per_node=1, the val shard and two training shards, until 20 training
steps have printed, then stops. It says nothing about speed or loss.

Checked on one H100 with two real candidates: multi-token-prediction-2 (an Inductor AssertionError,
16384 v.s. 16383, in its loss) fails in 30 s with that traceback; learnable-xsa-2 compiles in 440 s and passes
at step 27. Settings: SMOKE_STEPS (20), SMOKE_TIMEOUT_S (1800).
