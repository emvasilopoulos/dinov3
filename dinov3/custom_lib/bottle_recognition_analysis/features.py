import torch


def distance(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    cos_sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0),
                                                    vec2.unsqueeze(0)).item()
    return 1 - cos_sim
