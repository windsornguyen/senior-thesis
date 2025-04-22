def vocab_majority(
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
    end = 0

    inputs = []
    labels = []
    for _ in range(num_examples):

        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(1, vocab_size, size=input_seq_len-1)
        seq = [int(s) for s in seq]
        counter = Counter(seq)
        max_count = max(counter.values())
        keys_with_max_count = [k for k, v in counter.items() if v == max_count]
        most_key = min(keys_with_max_count)
        if len(keys_with_max_count) > 1:
            other_key = [k for k in counter.keys() if k != most_key][0]
            # replace an instance of other_key with most_key
            idx = seq.index(other_key)
            seq[idx] = most_key
        print(seq)

        # Full sequence
        input = np.array(seq + [end] + [most_key], dtype=np.int64)

        # Inputs and outputs
        input = torch.tensor(input)
        label = torch.full_like(input[:-1], -100) # -100 for labels, except last position
        label[-1] = input[-1]
        input = input[:-1]

        inputs.append(input)
        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    
    return DataSegment(
        inputs, 
        labels, 
        slices={"vocab_size": vocab_size}
    )