import torch

def compute_pairwise_distances(x, y):
    """
    Compute pairwise squared Euclidean distances between x and y.
    x: [n_samples_x, n_features]
    y: [n_samples_y, n_features]
    """
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return dist

    # x_norm = (x**2).sum(1).view(-1, 1)
    # if y is not None:
    #     y_t = torch.transpose(y, 0, 1)
    #     y_norm = (y**2).sum(1).view(1, -1)
    # else:
    #     y_t = torch.transpose(x, 0, 1)
    #     y_norm = x_norm.view(1, -1)
    
    # dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # # dist = torch.mm(x, y_t)
    # # # Ensure diagonal is zero if x=y
    # # if y is None:
    # #     dist = dist - torch.diag(dist.diag)
    # return torch.clamp(dist, 0.0, np.inf)


def MMD2(x, y, kernel="rbf", device="cpu"):
    """
    Empirical Maximum Mean Discrepancy (MMD) between x and y using specified kernel.
    Lower MMD indicates distributions are closer.
    Args:
        x: Tensor [N_x, D]
        y: Tensor [N_y, D]
        kernel: "rbf" or "multiscale"
        device: "cpu" or "cuda"
    Returns:
        Scalar MMD value
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2. * xx  # For K(x, x)
    dyy = ry.t() + ry - 2. * yy  # For K(y, y)
    dxy = rx.t() + ry - 2. * zz  # For K(x, y)

    XX = torch.zeros(xx.shape).to(device)
    YY = torch.zeros(yy.shape).to(device)
    XY = torch.zeros(zz.shape).to(device)

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx).pow(-1)
            YY += a**2 * (a**2 + dyy).pow(-1)
            XY += a**2 * (a**2 + dxy).pow(-1)

    elif kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    else:
        raise ValueError("Unsupported kernel type. Use 'rbf' or 'multiscale'.")

    mmd = torch.mean(XX + YY - 2. * XY)
    return mmd
