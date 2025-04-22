def cumulative_majority(
    config: CumulativeMajorityConfig,
    seed: int, 
    **kwargs
) -> DataSegment:
    """
    Generate majority sequences.
    """
    np.random.seed(seed)

    inputs = np.random.randint(0, 2, size=(config.num_examples, config.input_seq_len))
    labels = ((inputs * 2 - 1).cumsum(axis=1) >= 0).astype(int)

    return DataSegment(
        torch.tensor(inputs), 
        torch.tensor(labels), 
        slices={"input_seq_len": config.input_seq_len}
    )
