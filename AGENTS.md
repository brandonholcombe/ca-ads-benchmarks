# Benchmark repository rules

- This is the standalone NCA-ADS benchmark repository; read README.md and docs/BLACKWELL-SETUP.md.
- Never describe synthetic timing examples, CPU tests, syntax checks or mocks as GPU evidence.
- GPU mode must fail visibly when CUDA/cuDF/CuPy or a device is unavailable; never silently fall back.
- Validate results before publishing speedups. Keep correctness checks outside timers, sync GPU
  work around timings, and distinguish device-resident from host-to-host boundaries.
- Keep inputs deterministic and bounded. Preserve duplicate multiplicity, integer totals, float
  tolerances, raw samples, versions, hardware context and the source revision in reports.
- Preserve previous results and error evidence. Do not overwrite reports or commit generated runs,
  private data, tokens, environment files containing secrets, or credentials.
- No cloud provisioning, host-driver changes, instance resizing, purchases or paid APIs without
  explicit authorization. Existing GPU access is not permission to create additional resources.
- Use one allocated GPU for this first suite. Multi-GPU scaling needs a separately defined protocol.
- Run meaningful CPU tests and fail-path tests after changes. GPU validation remains pending until
  someone executes the exact revision on compatible hardware and returns the actual reports.
- Update the setup references when changing a GPU API or release. Keep the CPU baseline inside
  the same remote environment as GPU measurements. Do not claim packages are locked by a tag alone.
- The textbook and animation project live outside this repository. Proposed course claims should
  cite reviewed run artifacts; benchmark results do not authorize changing narration or video timing.
