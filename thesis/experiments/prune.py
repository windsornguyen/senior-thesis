import math


def lifetime(i):
    if i == 0:
        return 1
    k = int(math.log2(i & -i))  # Get the position of the rightmost set bit
    return 2 ** (k + 2) + 1


def is_alive(i, t):
    return i <= t <= i + lifetime(i) - 1


def pruning_algorithm(T):
    """
    Generalized pruning algorithm function.

    Args:
    T (int): The number of time steps to run the algorithm for.

    Returns:
    list: A list of sets, where the t-th set (0-indexed) represents S_{t}.
    """
    alive_sets = []
    for t in range(1, T + 1):
        S_t = set(i for i in range(1, t + 1) if is_alive(i, t))
        alive_sets.append(S_t)
    return alive_sets


def print_table(alive_sets):
    T = len(alive_sets)
    max_width = max(len(str(s)) for s in alive_sets)

    print(f"{'Time':5} | {'Alive Set':{max_width}} | New | Removed")
    print("-" * (max_width + 20))

    for t in range(1, T + 1):
        new = alive_sets[t - 1] - (alive_sets[t - 2] if t > 1 else set())
        removed = (alive_sets[t - 2] if t > 1 else set()) - alive_sets[t - 1]

        alive_set_str = ", ".join(map(str, sorted(alive_sets[t-1])))
        print(f"{t:5} | {alive_set_str:{max_width}} | {sorted(new)} | {sorted(removed)}")


def verify_property_1(alive_sets):
    for t, S_t in enumerate(alive_sets, start=1):
        for s in range(1, t + 1):
            interval = set(range(s, (s + t) // 2 + 1))
            assert S_t.intersection(interval), f"Property 1 violated at t={t}, s={s}"


def verify_property_2(alive_sets):
    for t, S_t in enumerate(alive_sets, start=1):
        if t == 1:
            assert len(S_t) == 1, f"Property 2 violated at t=1: |S_t| = {len(S_t)} != 1"
        else:
            assert len(S_t) <= 3 * math.log2(t) + 3, f"Property 2 violated at t={t}: |S_t| = {len(S_t)} > 3 * log2(t) + 3 = {3 * math.log2(t) + 3}"


def verify_property_3(alive_sets):
    for t in range(2, len(alive_sets) + 1):
        assert alive_sets[t - 1] - alive_sets[t - 2] == {t}, f"Property 3 violated at t={t}"


def prove_logarithmic_size(alive_sets):
    for t, S_t in enumerate(alive_sets, start=1):
        count = 0
        for k in range(math.floor(math.log2(t)) + 1):
            lower = max(1, t - 2 ** (k + 2) - 1)
            upper = t
            count_k = sum(
                1 for i in range(lower, upper + 1) if i % 2**k == 0 and i // 2**k % 2 == 1 and is_alive(i, t)
            )
            count += count_k
        assert count == len(S_t), f"Logarithmic size proof failed at t={t}"


# Set the number of time steps to simulate
T = 512

# Run the generalized pruning algorithm
alive_sets = pruning_algorithm(T)

# Print the table
print_table(alive_sets)

# Verify properties
verify_property_1(alive_sets)
verify_property_2(alive_sets)
verify_property_3(alive_sets)

# Prove logarithmic size
prove_logarithmic_size(alive_sets)

print("All properties verified and proofs passed!")
