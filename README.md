# Reproduce CUDA Graphs + CUPTI IMA issue

## Background
This issue appears in PyTorch: https://github.com/pytorch/pytorch/issues/75504; CUDA graph users find that if they:
* record & create a cudagraph
* begin the first profiling session
* run the cudagraph while profiling

... then, they will encounter an illegal memory access when they run the cudagraph. The minimal repro of this issue is shown in this repository: first we record the cudagraph; then we initialize CUPTI and register callbacks; then we run the cudagraph and encounter the illegal memory access.

Note, we only see this issue in older driver versions. On newer versions, we instead see the process hanging in the repro from pytorch/pytorch#75504. However, the hanging process does _not_ reproduce in this minimal repro.

For convenience, the repro for pytorch/pytorch#75504 is shown below:
```python
import os

import torch
from torch.profiler import ProfilerActivity, profile

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def add_one(in_: torch.Tensor):
    return in_ + 1


sample_arg = torch.zeros(10, device="cuda").requires_grad_(True)
add_one_graphed = torch.cuda.graphs.make_graphed_callables(add_one, sample_args=(sample_arg,))

zeros = torch.zeros(10, device="cuda")
out = add_one_graphed(zeros)
assert out[0] == 1

# This works
with profile(activities=[ProfilerActivity.CPU]):
    add_one_graphed(zeros)

# RuntimeError: CUDA error: an illegal memory access was encountered
with profile(activities=[ProfilerActivity.CUDA]):
    add_one_graphed(zeros)
```

## Usage
`bash build.sh`, then `./main`.  The script is fragile, so you'll probably need to modify it with all the appropriate paths for your CUDA installation.

## IMA issue (A100, CUDA 11.4, older drivers)
```
$ ./main
 time without cuda graph: 217245 us
 | initializing ProfilingState
 step 0
 step 1
Error: main.cu:106code:700, reason an illegal memory access was encountered
```

nvidia-smi (for driver info):
```
$ nvidia-smi
Wed Apr  5 11:59:02 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA PG509-210    On   | 00000000:11:00.0 Off |                    0 |
| N/A   41C    P0    92W / 330W |   1554MiB / 81251MiB |      0%      Default |
|                               |                      |             Disabled |
```

## Hanging process issue (A100, CUDA 11.6, newer drivers)
```
```
