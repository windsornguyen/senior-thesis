def linear_warmup_linear_decay(
    warmup_steps: int,
    decay_steps: int,
    current_step: int,
) -> float:
    """
    Computes linear warmup followed by linear decay, 
    following this paper: https://arxiv.org/pdf/2310.07831
    
    Per LambdaLR requirement, this is accomplished by returning a multiplicative
    factor to adjust the learning rate to create the desired schedule.
    """
    # linear decay schedule should have step size proportional to 1 - t/T, where t
    # current iter and T is total number of steps.
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence +1 adjustments
        current_step += 1
        current_adjustment = float(current_step / (warmup_steps + 1))
    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_steps)
        current_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    # is this returning the right thing?
    # TODO: Verify that this is correct (make a plot + look at initial LR values).
    return current_adjustment
