import random
import torch


def generate_ppf(points, reference='random'):
    '''
        points: (batch_size, num_points, 6)
    '''
    B, N = points.size(0), points.size(1)

    if reference == 'random':
        idx = random.choices(list(range(N)), k=B)
        target = torch.unsqueeze(points[torch.arange(B), idx], dim=1)  # (B, 1, 6)
    if reference == 'mean':
        mean_point = torch.sum(points, dim=1, keepdim=True)[:, :, :3]
        dist = torch.cdist(mean_point, points[:, :, :3]).squeeze()
        idx = torch.argmin(dist, dim=1)
        target = torch.unsqueeze(points[torch.arange(B), idx], dim=1)  # (B, 1, 6)
    return compute_ppf(target, points)
    

def compute_ppf(target, points):
    B, N = points.size(0), points.size(1)

    target_extend = target.repeat(1, N, 1)  # (B, N, 6)
    d = points[:, :, :3] - target_extend[:, :, :3]  # differnce vector, (B, N, 3)
    n1, n2 = target[:, :, 3:], points[:, :, 3:]  # surface normals, (B, 1, 3), (B, N, 3)

    dist = torch.cdist(target[:, :, :3], points[:, :, :3]).squeeze()  # euclidean distance, (B, N)
    angle_n1_d = torch.atan2(torch.linalg.norm(torch.cross(n1.repeat(1, N, 1), d), dim=2), torch.sum(n1 * d, dim=2))  # (B, N)
    angle_n2_d = torch.atan2(torch.linalg.norm(torch.cross(n2, d), dim=2), torch.sum(n2 * d, dim=2))  # (B, N)
    angle_n1_n2 = torch.atan2(torch.linalg.norm(torch.cross(n1.repeat(1, N, 1), n2), dim=2), torch.sum(n1 * n2, dim=2))  # (B, N)
    ppf = torch.stack([dist, angle_n1_d, angle_n2_d, angle_n1_n2], dim=-1)  # (B, N, 4)
    return ppf
