# Initial repository handoff

Prepared 2026-09-28 Pacific for the existing Akamai RTX PRO 6000 Blackwell host.

- Repository: [brandonholcombe/ca-ads-benchmarks](https://github.com/brandonholcombe/ca-ads-benchmarks), private, default branch `main`.
- Implementation: three CPU/GPU workloads, small correctness fixtures, two GPU timing boundaries,
  structured reports and failure reporting.
- Remote GPU execution: not performed; the first returned preflight and run artifacts establish it.
- Local validation: 18 CPU/failure-path tests passed; a 10,000/100,000-row run passed all six
  CPU cases. GPU preflight correctly failed on the Mac and retained an error JSON. Bash syntax
  and mocked Docker argument checks passed. See the implementation review for remaining limits.
- Clone the GitHub repository on the GPU host using your normal authorized GitHub access, then
  follow README.md and docs/BLACKWELL-SETUP.md. No token belongs in the repository or result files.

Only this benchmark directory belongs in this repository. Do not upload the surrounding
courseware, animation repository, local environments or historical videos. No license is assigned
by this setup; keep the repository private until the owner decides distribution terms.
