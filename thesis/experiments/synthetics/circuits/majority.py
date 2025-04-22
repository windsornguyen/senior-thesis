def majority(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate majority sequences.
    """
    np.random.seed(seed)

    one = 1
    zero = 0
    end = 2

    inputs = []
    labels = []
    # slices = []
    for _ in range(num_examples):
        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(0, 2, size=input_seq_len-1)
        seq = [one if s == 1 else zero for s in seq]
        num_ones = sum(seq)
        ratio = num_ones / len(seq)
        most = ratio > 0.5
        ratio = round(ratio * 5) / 5 # round ratio to nearest 0.2

        # Full sequence
        input = np.array(seq + [end] + [most], dtype=np.int64)

        # Inputs and outputs
        input = torch.tensor(input)
        label = torch.full_like(input[:-1], -100) # -100 for labels, except last position
        label[-1] = input[-1]
        input = input[:-1]

        inputs.append(input)
        labels.append(label)
        # slices.append(ratio)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    
    return DataSegment(
        inputs, 
        labels, 
        # slices={"ratio": slices}
    )
    