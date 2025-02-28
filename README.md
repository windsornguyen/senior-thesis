# Senior Thesis

Source code for my senior thesis at Princeton University.

## Checklist
Things left to do:

- [] Do we need the global_rank for the nd-dataloader?
- [] Fix logger distributed import circular problem
- [] Check if we still need optimizers.py after build_optimizers.py
- [] Make nd_dataset production grade
   - [ ] **Shard Shuffling**: Ensure shards (and possibly token chunks) are randomized each epoch to avoid training bias.  
   - [ ] **Memory Mapping / Streaming**: Avoid loading entire shards into memory (use `np.memmap` or a similar streaming approach).  
   - [ ] **Parallel I/O**: Enable multi-process or multi-thread I/O (e.g., `num_workers` in DataLoader) and consider pinned memory for faster data transfer.  
   - [ ] **Epoch Boundaries**: Provide explicit epoch control rather than infinite looping, so you can accurately count passes over the data.  
   - [ ] **2D Partitioning Logic**: If dimension 1 is also for data slicing, revise shard assignment to include both `dp_rank` and `tp_rank`.  
   - [ ] **Advanced Checkpointing**: Store and reload iteration state robustly for partial shards, ensuring it works across all parallel ranks if needed.  
   - [ ] **Error Handling & Logging**: Improve error messages for corrupted files, dimension mismatches, or out-of-range shard indices, and log essential info at each rank.

### Low priority
- [] Inference wrapper, ideally wrapped during training on master process to check model progress
- [] Add context parallelism for non-Attention models, if possible, e.g. Mamba
- [] Fix logger ([rank0]:[rank0]:, ...), also no color
- [] Setup WandB
- [] Integrate Kilian's code for offline WandB sync
- [] can't use torchrun for some reason (comm)