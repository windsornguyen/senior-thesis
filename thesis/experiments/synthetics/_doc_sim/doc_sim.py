import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def generate_document_similarity(
    num_examples: int = 10,
    num_documents: int = 10,
    num_elements: int = 10,
    top_k: int = 2,
    seed: int = 1_337,
    dtype: torch.dtype = torch.bfloat16,
) -> TensorDataset:
    """
    Generate a dataset for the cosine similarity task. The goal is to find the
    pair of documents (tensors) with the highest cosine similarity.
    
    Alman and Yu (2024) claimed that no sub-quadratic model can solve this task:
    https://arxiv.org/abs/2410.04271.
    
    
    Args:
        num_examples (int): Number of examples (sets of documents) to generate.
        num_documents (int): Number of documents (tensors) in each example.
        num_elements (int): Number of elements in each document.
        top_k (int): Number of top similar document pairs to identify.
        seed (int): Random seed for reproducibility.
        dtype (torch.dtype): Data type for the tensors.
    
    Returns:
        TensorDataset:
            - Inputs: Shape (num_examples, num_documents, num_elements)
            - Targets: Shape (num_examples, top_k, 2) - the indices of the document pairs 
              with the highest cosine similarity.
    """
    torch.manual_seed(seed)

    # Validate parameters
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if num_documents < 2:
        raise ValueError("num_documents must be at least 2 to form pairs.")
    max_topk = num_documents * (num_documents - 1) // 2
    if top_k > max_topk:
        raise ValueError(f"top_k={top_k} exceeds the maximum number of unique document pairs ({max_topk}).")
    
    # Generate and normalize the inputs
    inputs = torch.randn((num_examples, num_documents, num_elements), dtype=dtype)
    normalized_inputs = F.normalize(inputs, p=2, dim=2)
    
    # Compute the cosine similarity between all pairs of documents
    cosine_similarity = normalized_inputs @ normalized_inputs.transpose(1, 2)
    
    # Get upper triangular indices (excluding diagonal)
    triu_indices = torch.triu_indices(num_documents, num_documents, offset=1)
    
    # Extract upper triangular similarities
    sim_pairs = cosine_similarity[:, triu_indices[0], triu_indices[1]]  # Shape: (num_examples, num_pairs)
    
    # Get top_k indices for each example
    topk_values, topk_indices = torch.topk(sim_pairs, top_k, dim=1, largest=True, sorted=True)
    
    # Map topk_indices to pair indices
    topk_pairs = triu_indices[:, topk_indices]  # Shape: (2, num_examples, top_k)
    targets = topk_pairs.permute(1, 2, 0)  # Shape: (num_examples, top_k, 2)

    return TensorDataset(inputs, targets)
