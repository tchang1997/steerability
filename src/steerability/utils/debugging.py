import torch 

from typing import Optional

def _multiprocess_breakpoint(rank_to_debug: Optional[int] = 0):
    # Note that this is not equivalent to fully multi-processed debugging + can't debug comm. across processes — this is only for making sure *one* worker functions OK
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0 

    if rank == rank_to_debug:
        import pdb; pdb.set_trace() # HOOOOOH MULTI PROCESS DEBUGGING
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier() # this will cause an error if we `cont` in a single-process env.