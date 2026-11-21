# DataSci GPU cluster investigation

**Last live inspection: 2026-11-21, 20:41–20:44 UTC (15:41–15:44 America/Toronto).**

Running investigation for the SE390 kernel benchmarking project. Facts below are either live observations, earlier email statements, documentation claims, or explicitly labeled interpretations. Cluster availability can change immediately; these are not benchmark results.

**Follow-up:** the [compute CLI implementation and live-test report](compute-cli-validation.md) records the subsequent PyTorch environment installation, successful CUDA compilation/execution on `gpu-8-vm`, profiling results, and failure/cancellation tests. Statements below about packages not being installed or kernels not being executed refer to the original inventory inspection, not the later implementation work.

## Outcome

SSH authentication succeeded as `ikorovin` to `datasci-gpu.cs.uwaterloo.ca` (login host: `slurm-login`). Four short Slurm inspection jobs successfully ran on `gpu-8-vm`, `gpu-1-vm`, and `gpu-6-vm`; all completed with exit code `0:0`. The final user queue was empty.

- **`gpu-8-vm` is usable and was idle:** a complete NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, MIG disabled, 97,887 MiB visible GPU memory (about 95.6 GiB), compute capability 12.0.
- **`gpu-1-vm` is also non-MIG and was idle:** same GPU model and memory, but a different default CUDA Toolkit version.
- **`gpu-6-vm` had a free `2g.48gb` MIG device:** this is a partition, not the whole 96 GB-class physical GPU.
- **Driver on all three inspected nodes:** `595.58.03`; `nvidia-smi` reports CUDA `13.2`. Installed compiler versions were checked separately below.
- The live scheduler has seven nodes and eight GPU **resource units**. These units must not be counted as eight physical GPU cards; MIG devices can be separately scheduled.

## Live availability snapshot

Source: `sresources`, corroborated by `scontrol show nodes`, and refreshed after the inspection jobs at 20:44 UTC. Memory in this table is **host RAM**, not GPU VRAM. Free RAM/CPU values are scheduler capacity, not instantaneous OS memory usage or a per-user entitlement.

| Node | Slurm state | GPU units total / allocated / free | CPU cores total / allocated / free | Host RAM total / allocated / available | Direct GPU inspection |
| --- | --- | --- | --- | --- | --- |
| `gpu-1-vm` | IDLE | 1 / 0 / 1 | 32 / 0 / 32 | 186.7G / 0 / 186.7G | Full RTX PRO 6000 Blackwell Max-Q; MIG off |
| `gpu-2-vm` | MIXED | 1 / 1 / 0 | 32 / 16 / 16 | 186.7G / 96G / 90.7G | Not inspected: GPU resource occupied |
| `gpu-3-vm` | MIXED | 1 / 1 / 0 | 32 / 16 / 16 | 186.7G / 170G / 16.7G | Not inspected: GPU resource occupied |
| `gpu-4-vm` | MIXED | 1 / 1 / 0 | 32 / 16 / 16 | 186.7G / 170G / 16.7G | Not inspected: GPU resource occupied |
| `gpu-5-vm` | MIXED | 1 / 1 / 0 | 32 / 16 / 16 | 186.7G / 170G / 16.7G | Not inspected: GPU resource occupied |
| `gpu-6-vm` | MIXED | 2 / 1 / 1 | 32 / 8 / 24 | 263.9G / 64G / 199.9G | Allocated free `2g.48gb` MIG device |
| `gpu-8-vm` | IDLE | 1 / 0 / 1 | 32 / 0 / 32 | 186.7G / 0 / 186.7G | Full RTX PRO 6000 Blackwell Max-Q; MIG off |

`gpu-7-vm` was not present in the node or partition inventory. The model, driver, toolkit, and MIG status of nodes 2–5 have not been established by this investigation. No attempt was made to access their occupied GPU resources outside Slurm.

## GPU and CUDA configuration

Source: `nvidia-smi`, `/usr/local/cuda/bin/nvcc --version`, and CUDA symlink inspection inside successful scheduler allocations.

| Property | `gpu-8-vm` | `gpu-1-vm` | `gpu-6-vm` allocated device |
| --- | --- | --- | --- |
| Physical GPU model | RTX PRO 6000 Blackwell Max-Q Workstation Edition | Same | Same backing GPU model |
| MIG current / pending | Disabled / Disabled | Disabled / Disabled | Enabled / Enabled |
| Allocated GPU resource | Whole GPU | Whole GPU | `2g.48gb` MIG instance |
| GPU memory | 97,887 MiB visible | 97,887 MiB visible | 48 GB-class MIG profile; the physical-device query reports 97,887 MiB, which is **not** this job's memory entitlement |
| Compute capability | 12.0 | 12.0 | Backing GPU reports 12.0 |
| NVIDIA driver | 595.58.03 | 595.58.03 | 595.58.03 |
| `nvidia-smi` CUDA version | 13.2 | 13.2 | 13.2 |
| Installed `nvcc` | 13.2, V13.2.86 | 13.3, V13.3.73 | 13.2, V13.2.86 |
| `/usr/local/cuda` resolves to | `/usr/local/cuda-13.2` | `/usr/local/cuda-13.3` | `/usr/local/cuda-13.2` |
| Physical GPU power limit | 300 W | 300 W | 300 W backing-GPU limit, not a dedicated MIG power budget |
| Persistence / compute mode | Enabled / Default | Enabled / Default | Enabled / Default on backing GPU |

The CUDA version reported by `nvidia-smi` describes the driver's CUDA support, not the installed toolkit. `gpu-1-vm` demonstrates the distinction: driver-reported 13.2 and installed compiler 13.3. No compilation or runtime compatibility test was performed, so this observation is not a claim that this combination will or will not run a particular workload.

Neither `nvcc` nor `conda` was on PATH in the noninteractive job environments inspected. The CUDA compiler **is installed** and executable by its absolute path. This does not establish whether interactive shell initialization or other user environments expose additional tools. Python resolved to `/usr/bin/python3`.

On `gpu-8-vm`, additional observations were:

- OS: Ubuntu 24.04.4 LTS; Slurm node kernel: `6.8.0-149-generic`.
- CPU: AMD EPYC 9354P 32-Core Processor, 32 visible logical CPUs, one thread per core.
- Slurm-configured host RAM: `191204M` (about 186.7 GiB).
- GPU power range reported: 250–325 W, current/default limit 300 W. Reported maximum SM clock 3090 MHz and memory clock 14001 MHz are not measured sustained clocks.
- VM passthrough comes from the earlier administrator email; no host-level virtualization configuration was independently inspected.

## Scheduler and resource limits

Sources: `scontrol show partition`, selected fields from `scontrol show config`, `sacctmgr`, and submission dry runs.

| Setting | Observed value | Interpretation / limitation |
| --- | --- | --- |
| Slurm version | 24.11.6 | Version reported by client and node records |
| Partition | `ALL`, default, UP | All seven visible compute nodes belong to it |
| Partition `MaxTime` | `7-00:00:00` | Configured seven-day maximum; runtime enforcement was not tested |
| Interactive limit | Not independently established | Public docs previously said seven hours; visible partition configuration alone does not prove a separate interactive cap |
| Total cluster capacity | 224 CPU cores, 1,417,428M host RAM, 8 GPU resource units | Global capacity, not a per-user allocation guarantee |
| Preemption | `PreemptMode=OFF`, `PreemptType=(null)` | Matches administrator's statement that running jobs are not preempted |
| Oversubscription | `OverSubscribe=NO` | Do not infer whole-node exclusivity from receiving one GPU |
| Resource selection | `select/cons_tres`, `CR_CORE_MEMORY` | CPU/memory resources are scheduled |
| Isolation | `task/cgroup,task/affinity`; `proctrack/cgroup` | Task and process isolation configured |
| Account/association enforcement | `AccountingStorageEnforce=none` | No accounting association enforcement configured in the reported setting |
| Own association / user records | Queries returned no data rows | No visible per-user association limits; this is not proof of unlimited use or permission |
| QoS | `normal`, limit fields blank | No explicit caps shown in the queried QoS; administrative policies can still apply |
| Partition maximum nodes/CPUs/memory | Listed as UNLIMITED | Still bounded by physical resources, scheduler availability, and applicable policies |
| Job submission plugin | `(null)` | No submit plugin listed in the inspected configuration |
| Maximum job count | 10,000 | Scheduler-wide setting, not a personal quota |
| Maximum array size | 1,001 | Scheduler configuration value, not a tested user allowance |
| Storage | Home filesystem reports 1.0T total and available | Filesystem-reported capacity is **not** a verified personal storage quota; `quota` was not on PATH |
| Login-shell process limits | Open files 1,024; stack 8,192 KiB; core dumps disabled | Login shell only; job-specific process limits were not checked |

### Submission validation caveat

Two `sbatch --test-only` requests were made; neither submitted a persistent job nor reserved compute resources:

1. `gpu-8-vm`, 32 CPU cores, `191204M` host RAM, one GPU, seven days: exit 0; predicted start `2026-11-22T20:44:04` using 32 processors.
2. Same node, one CPU, 1G RAM, one GPU, **seven days plus one second**: also exit 0 with the same predicted start.

The second dry run did **not** reject the over-limit duration. Thus these dry runs cannot be used as proof of wall-time enforcement or guaranteed permission to run beyond the partition maximum. The configured maximum remains seven days; ask admins if a boundary guarantee is needed. A predicted start is only a scheduler estimate, not a booking. Short inspection jobs started successfully even though the long dry runs predicted a later start.

## Implications for kernel benchmarking

These are project interpretations of the observations, not measured performance results:

- `gpu-8-vm` is a practical target for full-GPU experiments because it was successfully allocated, is non-MIG, and has an explicitly verified CUDA 13.2 compiler.
- `gpu-1-vm` is another full-GPU option, but its default toolkit differs. Pin the compiler/runtime environment before pooling results across nodes.
- On the inspected non-MIG nodes, `--gres=gpu:1` yielded a complete GPU. On `gpu-6-vm`, the same request yielded a `2g.48gb` partition. The earlier documentation's generic quarter-GPU description is not a reliable universal rule for the current cluster.
- A multi-slice allocation does not itself disable MIG. Treat slice and full-GPU measurements as different experimental conditions.
- Record the exact Max-Q GPU variant, VM setting, MIG profile, driver/toolkit, power limit, node, and Slurm allocation alongside results. Do not assume equivalence to other RTX PRO 6000 editions or bare-metal benchmark scores.
- Receiving a GPU allocation does not guarantee exclusive host CPU/memory use. The inspection jobs requested one CPU and 1G host RAM; those requests were sufficient only for metadata inspection, not established benchmark requirements.
- No GPU kernel was compiled, launched, or timed as a performance test. Python framework installations, cuDNN, Triton, and PyTorch compatibility were not checked.

## Reproducing the inspection

Connect with your registered SSH key:

```bash
ssh ikorovin@datasci-gpu.cs.uwaterloo.ca
```

Read scheduler state without allocating a GPU:

```bash
sresources
scontrol show nodes
scontrol show partition
squeue -u ikorovin
sacctmgr -P show qos
```

For a brief, scheduler-managed inspection of `gpu-8-vm`, the following uses one GPU, one CPU and 1G host RAM, caps runtime at one minute, and gives up if resources cannot be allocated promptly:

```bash
srun --immediate=10 --partition=ALL --nodelist=gpu-8-vm \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G \
  --gres=gpu:1 --time=00:01:00 --job-name=se390-inventory \
  bash -c '
    hostname
    nvidia-smi --query-gpu=name,memory.total,driver_version,mig.mode.current,compute_cap --format=csv
    nvidia-smi --version
    /usr/local/cuda/bin/nvcc --version
  '
```

This compact command reproduces the main checks; the original jobs used a Python subprocess wrapper and additionally read device enumeration, power limits, symlinks, and OS/CPU details. Use an explicit toolkit path in jobs, or set the job's environment deliberately. No shell startup files or packages were modified during this investigation.

## Access and cleanup audit

- Initial strict host-key verification stopped because this hostname had no known-hosts entry.
- Authentication then succeeded using SSH's `accept-new` policy. This recorded a previously unknown host key and continued to reject changed known keys. It was trust on first use; the fingerprint was not independently matched against an administrator-published value.
- Later probes used strict host-key verification. Automatic additional host-key updates were disabled for subsequent parallel/probe connections after an update-file warning; verification of the recorded host key remained enabled.
- No passwords or private keys were read into the report. No packages were installed, GPU modes changed, or administrator settings modified.
- Jobs `8109` and `8110` inspected `gpu-8-vm`; `8111` inspected `gpu-1-vm`; `8112` inspected the free allocation on `gpu-6-vm`. Each requested one CPU, one GPU resource and 1G RAM, with a one- or two-minute maximum.
- All four jobs completed with exit code `0:0`. Reported elapsed times were 0, 1, 2 and 0 seconds respectively (Slurm's second-resolution values).
- The final `squeue -h -u ikorovin` output was empty. No investigation allocations were left running or queued.

## Previously established email context


Source: Outlook thread **“Access to Datasci GPU Cluster: SE390”**, messages dated October 16–22, 2026, inspected on November 21. These email observations are separate from live cluster checks:

- Josiah Taylor confirmed account creation for the three project members on October 19, with one-year expiry and the possibility of extension. Exact account expiration was not supplied.
- GPUs were passed through to VMs. Josiah described a crude estimate of 5–10% performance loss; this is not a measured result from this project.
- Most GPUs used MIG. Josiah identified `gpu-8-vm` as non-MIG and offered to configure another non-MIG node if required.
- On October 22, Indy Ng said a lingering process on `gpu-8-vm` had been cleared and the node returned to service. Jobs queue without preempting running jobs; Indy said there was no whole-node reservation system to his knowledge.
- Administrators requested reports of nodes showing `IDLE/DRAINED`. Support: `dl-datasci-admin@uwaterloo.ca`.

## Evidence excerpts

The following outputs were captured from live commands during this investigation. Device UUIDs and the internal NFS endpoint are omitted because they are unnecessary for interpreting the findings. Source commands and job/node identities are retained.

### E1: resource availability after cleanup

Command: `sresources`

```text
Gathering Slurm resource data from nodes...
NODE      STATE  GPUType  GPUTot  GPUAlloc  GPUFree     MEM  AllocMEM  FreeMEM  CPUTot  CPUAlloc  CPUFree
---------------------------------------------------------------------------------------------------------
gpu-1-vm  IDLE   gpu           1         0        1  186.7G        0M   186.7G      32         0       32
gpu-2-vm  MIXED  gpu           1         1        0  186.7G       96G    90.7G      32        16       16
gpu-3-vm  MIXED  gpu           1         1        0  186.7G      170G    16.7G      32        16       16
gpu-4-vm  MIXED  gpu           1         1        0  186.7G      170G    16.7G      32        16       16
gpu-5-vm  MIXED  gpu           1         1        0  186.7G      170G    16.7G      32        16       16
gpu-6-vm  MIXED  gpu           2         1        1  263.9G       64G   199.9G      32         8       24
gpu-8-vm  IDLE   gpu           1         0        1  186.7G        0M   186.7G      32         0       32
```

### E2: partition configuration

Command: `scontrol show partition`

```text
PartitionName=ALL
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=YES QoS=N/A
   DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=NO ExclusiveTopo=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=7-00:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
   NodeSets=ALL
   Nodes=gpu-1-vm,gpu-2-vm,gpu-3-vm,gpu-4-vm,gpu-5-vm,gpu-6-vm,gpu-8-vm
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=224 TotalNodes=7 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED
   TRES=cpu=224,mem=1417428M,node=7,billing=224,gres/gpu=8
```

### E3: gpu-8-vm GPU and toolkit

Captured 2026-11-21T20:43:17.196728+00:00, Slurm job `8110`.

Command: `nvidia-smi --query-gpu=index,name,memory.total,driver_version,mig.mode.current,mig.mode.pending,compute_cap --format=csv`

```text
index, name, memory.total [MiB], driver_version, mig.mode.current, mig.mode.pending, compute_cap
0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 595.58.03, Disabled, Disabled, 12.0
```

Command: `nvidia-smi --version`

```text
NVIDIA-SMI version  : 595.58.03
NVML version        : 595.58
DRIVER version      : 595.58.03
CUDA Version        : 13.2
```

Command: `/usr/local/cuda/bin/nvcc --version`

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Fri_May_08_10:53:34_AM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.86
Build cuda_13.2.r13.2/compiler.37953736_0
```

### E4: gpu-1-vm GPU and toolkit

Captured 2026-11-21T20:43:19.341389+00:00, Slurm job `8111`.

Command: `nvidia-smi --query-gpu=index,name,memory.total,driver_version,mig.mode.current,mig.mode.pending,compute_cap --format=csv`

```text
index, name, memory.total [MiB], driver_version, mig.mode.current, mig.mode.pending, compute_cap
0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 595.58.03, Disabled, Disabled, 12.0
```

Command: `nvidia-smi --version`

```text
NVIDIA-SMI version  : 595.58.03
NVML version        : 595.58
DRIVER version      : 595.58.03
CUDA Version        : 13.2
```

Command: `/usr/local/cuda/bin/nvcc --version`

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Jun_09_02:43:40_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.73
Build cuda_13.3.r13.3/compiler.38244171_0
```

### E5: gpu-6-vm GPU and toolkit

Captured 2026-11-21T20:43:20.418463+00:00, Slurm job `8112`.

Command: `nvidia-smi --query-gpu=index,name,memory.total,driver_version,mig.mode.current,mig.mode.pending,compute_cap --format=csv`

```text
index, name, memory.total [MiB], driver_version, mig.mode.current, mig.mode.pending, compute_cap
0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 595.58.03, Enabled, Enabled, 12.0
```

Command: `nvidia-smi --version`

```text
NVIDIA-SMI version  : 595.58.03
NVML version        : 595.58
DRIVER version      : 595.58.03
CUDA Version        : 13.2
```

Command: `/usr/local/cuda/bin/nvcc --version`

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Fri_May_08_10:53:34_AM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.86
Build cuda_13.2.r13.2/compiler.37953736_0
```

`nvidia-smi -L` additionally identified the allocated device as `MIG 2g.48gb Device 0`; `CUDA_VISIBLE_DEVICES` contained a MIG UUID.

### E6: selected scheduler configuration

Command: `scontrol show config` (filtered to the relevant fields below).

```text
AccountingStorageEnforce = none
AccountingStorageTRES   = cpu,mem,energy,node,billing,fs/disk,vmem,pages,gres/gpu,gres/gpumem,gres/gpuutil
AccountingStorageType   = accounting_storage/slurmdbd
DefMemPerNode           = UNLIMITED
JobSubmitPlugins        = (null)
MaxArraySize            = 1001
MaxJobCount             = 10000
MaxMemPerNode           = UNLIMITED
PreemptType             = (null)
PriorityType            = priority/multifactor
ProctrackType           = proctrack/cgroup
SchedulerParameters     = (null)
SchedulerType           = sched/backfill
SelectType              = select/cons_tres
SelectTypeParameters    = CR_CORE_MEMORY
TaskPlugin              = task/cgroup,task/affinity
```

### E7: inspection job completion

Command: `sacct -j 8109,8110,8111,8112 -P --format=JobID,JobName,NodeList,State,Elapsed,AllocCPUS,ReqMem,ExitCode`

```text
JobID|JobName|NodeList|State|Elapsed|AllocCPUS|ReqMem|ExitCode
8109|se390-inventory|gpu-8-vm|COMPLETED|00:00:00|1|1G|0:0
8109.extern|extern|gpu-8-vm|COMPLETED|00:00:00|1||0:0
8109.0|se390-inventory|gpu-8-vm|COMPLETED|00:00:00|1||0:0
8110|se390-inventory|gpu-8-vm|COMPLETED|00:00:01|1|1G|0:0
8110.extern|extern|gpu-8-vm|COMPLETED|00:00:01|1||0:0
8110.0|se390-inventory|gpu-8-vm|COMPLETED|00:00:00|1||0:0
8111|se390-inventory|gpu-1-vm|COMPLETED|00:00:02|1|1G|0:0
8111.extern|extern|gpu-1-vm|COMPLETED|00:00:02|1||0:0
8111.0|se390-inventory|gpu-1-vm|COMPLETED|00:00:00|1||0:0
8112|se390-inventory|gpu-6-vm|COMPLETED|00:00:00|1|1G|0:0
8112.extern|extern|gpu-6-vm|COMPLETED|00:00:00|1||0:0
8112.0|se390-inventory|gpu-6-vm|COMPLETED|00:00:00|1||0:0
```

## Open questions / next checks

- Confirm actual storage quota, retention/backup policy, and account expiration with administrators if these affect experiment planning.
- Confirm whether a separate seven-hour interactive limit is enforced outside the visible partition settings.
- Validate the selected project environment with a small correctness test before running performance experiments; no compatibility or timing tests were part of this inspection.
- Inspect nodes 2–5 only when their GPUs can be allocated normally, if a complete physical hardware inventory becomes necessary.
- Recheck scheduler availability immediately before experiments. Idle status and dry-run start estimates are not reservations.

## Public reference documentation

- [DataSci GPU overview](https://datasci-gpu.cs.uwaterloo.ca/)
- [Slurm and GPU resources](https://datasci-gpu.cs.uwaterloo.ca/slurm.html)
- [SSH / VS Code connection guide](https://datasci-gpu.cs.uwaterloo.ca/vscode.html)
- [Virtual environments and CUDA tools](https://datasci-gpu.cs.uwaterloo.ca/VirtualEnv.html)

These live documentation URLs were reviewed during the preceding email investigation. They are not frozen documentation snapshots. The live machine observations above take precedence for the configurations actually inspected; documentation discrepancies are recorded rather than silently reconciled.

Return to the [artifacts index](README.md).
