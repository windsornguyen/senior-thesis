def cumulative_parity(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate parity sequences.
    """
    np.random.seed(seed)

    one = 1
    zero = 0
    end = 2

    inputs = []
    labels = []
    for _ in range(num_examples):
        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(0, 2, size=input_seq_len-1)
        parities = []
        cur_parity = 0
        for i in range(input_seq_len-1):
            cur_parity = (cur_parity + seq[i]) % 2
            parities.append(cur_parity)

        # Append the parity to the sequence
        input = np.array(seq, dtype=np.int64)
        input = torch.tensor(input)
        label = torch.tensor(parities)

        inputs.append(input)
        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    
    return DataSegment(
        inputs, 
        labels, 
        slices={"input_seq_len": input_seq_len}
        # slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
    )

