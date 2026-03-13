# Autonomous LLM inference optimization research
You are tasked with autonomously improving the inference throughput of a fixed model: **Qwen3-30B-A3B-Instruct-2507 (GGUF Q4_K_M)**, running via llama-cpp-python. You reside on a GPU cluster and will execute experiments entirely unsupervised.

## Setup
To set up a new experiment, work with the user to:
1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read all (non-hidden) files in the `src/` directory.
4. **Verify that the environment exists**: Check that `.venv` exists. If not, run their installation scripts before beginning any experimentation.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation
Each experiment runs on a single Nvidia A100 (40GB) GPU with access to 16 CPU cores. The script runs for a **fixed time budget of 10 minutes** (wall clock time, including startup/compilation). You launch it simply as: `sbatch src/01_run.job`.
* Note: sometimes it takes a bit of time for the job to *actually start*, as there might be a queue for the GPUs. This does NOT count towards the 10 minutes.

### What you CAN do
- Modify `src/inference.py` — this is the only file you edit. Specifically, you may tune any llama-cpp-python parameters, inference settings, and runtime optimizations that affect throughput of the fixed model.
- Install new dependencies (**15 minute limit!**). Just update `src/requirements.txt` and sbatch `src/00_install_env.job`. Make sure to not let pip use cached packages and be careful with dependencies overwriting existing versions.
- Search the web for (recent) information and documentation on llama.cpp performance tuning.

**Suggested directions** (in rough priority order):

1. **Speculative decoding**: Use a small draft model (e.g. a Qwen3 0.6B or 1.7B GGUF) to speculatively generate tokens that the main model verifies in parallel. Reference: https://www.reddit.com/r/LocalLLaMA/comments/1hqlug2/revisting_llamacpp_speculative_decoding_w/ — use `llama_cpp`'s built-in speculative decoding support if available, or experiment with draft model parameters.

2. **CUDA Graphs + FlashAttention**: Enable CUDA graph capture and flash attention in llama.cpp for reduced kernel launch overhead and faster attention computation. Reference: https://github.com/ggml-org/llama.cpp/issues/6763 — may require recompiling llama-cpp-python with specific CUDA flags.

3. **Batch and threading parameters**: Tune `n_batch`, `n_ubatch`, `n_threads`, and `n_threads_batch` for the A100 + 16-core CPU setup. Well-chosen values can yield 10–25% faster prefill and 5–15% faster decode with zero code complexity cost — try these first before more invasive changes.

### What you CANNOT do
- **Change the main model.** The model is fixed: `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` at the path already defined in `inference.py`. Do not swap it for a different model or quantization level. Exception: you may use a *smaller* model as a **draft model** for speculative decoding only.
- Change the prompt the LLMs are running on. It is defined in `src/prompt.py`; changing this file disqualifies a run.
- Modify the evaluation harness. **Every run must write** `{"total_tokens": int, "generation_time": float, "tokens_per_second": float}` to `src/outputs/<job_id>.json`. A lack of these results results in the run receiving a score of 0.
- Note that "generation_time" is supposed to measure the time it takes for the model to process the tokens (input prompt and reasoning+output); it excludes (model) loading times.

**The goal is simple**: Get the highest score possible, where the score is defined as "tokens_per_second".

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.
* One more token/second that adds 40 lines of hacky code => probably not worth it.
* One more token/second from deleting code and/or dependencies => definitely keep.
* No improvement but much simpler code and/or dependencies => definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the script as is.

## Output format
After each run, a new file will be created in `src/outputs/<job_id>.json` that summarizes the results. It will be displayed at the end of a run.

## Logging results
When an experiment is done, log it to `src/outputs/<tag>.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:
```
commit	tk_s	status	description
```
1. git commit hash (short, 7 chars)
2. tokens_per_second achieved (e.g. 12.23) — use 0 for crashes/timeouts
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:
```
commit	tk_s	status	description
a1b2c3d	12.23	keep	baseline
c3d4ef1	14.72	keep	n_batch=512 n_ubatch=128
d8e7c1a	26.23	keep	speculative decoding with Qwen3-0.6B draft
c3d4e5f	0.000	crash	cuda graphs (compile timeout)
```

## Interacting with the GPU
The cluster uses a Slurm workload manager. All scripts run through a `.job` file specifying partition, timeout, number of tasks and more. **You are NOT to modify this**; the research goal is to maximize performance with this setup.

### Useful commands
All of these assume you are running from `/src`.
* `sbatch <file>.job` for running jobs, also returns `job_id`.
* `squeue` for checking the queue. Allows you to see when a job goes from being queued to running.
* `jlog <job_id>` (defined in `~/.bashrc`) monitors the structured log for a running job and exits when the job finishes.
* `python3 summarize_hw.py outputs/hw_monitor_<job_id>.csv` for a hardware utilization summary (after a run finishes).
* `python3 evaluate.py <job_id>` for an evaluation summary of the results (after a run finished).

### Logging
`inference.py` uses a logger (defined in `src/logger.py`) that writes structured lines — env info, model load time, generation stats — to both stdout and `src/outputs/<job_id>.log`. The raw LLM response is printed to stdout only and does not appear in the log file. Use `jlog` to monitor a run without the response flooding your context.

## The experiment loop
The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

**LOOP FOREVER**:

1. Look at the git state: the current branch/commit we're on.
2. Tune `src/inference.py` with an experimental idea by directly hacking the code.
3. `git commit` your changes.
4. Run the experiment: `sbatch src/01_run.job`.
5. Monitor the run: `jlog <job_id>`, do NOT monitor or read the raw (stdout) of runs, do NOT let redundant information flood your context.
6. Check the results in `outputs/<job_id>.json` and record them in the `.tsv` file.
7. If the score (tokens/s) improves, you 'advance' the branch by keeping the commit.
8. If the score (tokens/s) decreases, revert the changes you made to the code (keep record of results!).

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**GPU allocation**: If it takes a lot of time for the job to get allocated a GPU, there is NOTHING YOU CAN DO. It's the unfortunate truth, your best option is to keep running `jlog <job_id>` until it eventually gets scheduled.

**Timeout**: Each experiment should take ~10 minutes total. If a run exceeds 10 minutes IT WILL CRASH.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — research online, re-read the in-scope files for new angles, try combining previous near-misses, try more radical tuning combinations. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 50 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

GOOD LUCK!
