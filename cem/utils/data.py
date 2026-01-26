import torch
import numpy as np



def _largest_divisor(x, max_val):
    largest = 1
    for i in range(1, max_val + 1):
        if x % i == 0:
            largest = i
    return largest

def daloader_to_memory(
    dl,
    as_torch=False,
    num_workers=5,
    max_val=512,
    output_groups=False,
    only_labels=False,
    fast_loader=None,
    prefetch_factor=2,
    output_latent_concepts=False,
):
    c_latent_data = None
    if hasattr(dl.dataset, 'tensors'):
        x_data, y_data, c_data = dl.dataset.tensors[:3]
        if isinstance(c_data, (list, tuple)):
            # Then we were provided with a set of training concepts and a
            # set of latent concepts. Let's just use the training concepts
            # for evaluation
            c_data, c_latent_data = c_data[0], c_data[1]
        if output_groups:
            if len(dl.dataset.tensors) >= 4:
                g_data = dl.dataset.tensors[3]
            else:
                g_data =  np.ones((x_data.shape[0], 1), dtype=np.float32)
        if not as_torch:
            x_data = x_data.detach().cpu().numpy()
            y_data = y_data.detach().cpu().numpy()
            c_data = c_data.detach().cpu().numpy()
            if c_latent_data is not None:
                c_latent_data = c_latent_data.detach().cpu().numpy()
            if output_groups and (not isinstance(g_data, np.ndarray)):
                g_data = g_data.detach().cpu().numpy()
    else:
        if fast_loader is None:
            fast_loader = torch.utils.data.DataLoader(
                dl.dataset,
                batch_size=_largest_divisor(len(dl.dataset), max_val=max_val),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
        y_data, c_data, c_latent_data, g_data = [], [], [], []
        if not only_labels:
            x_data = []
        for data in fast_loader:
            c_latent = None
            if len(data) == 2:
                x, (y, c) = data
            else:
                (x, y, c) = data[:3]
            if isinstance(c, (list, tuple)):
                # Then we were provided with a set of training concepts and a
                # set of latent concepts. Let's just use the training concepts
                # for evaluation
                c, c_latent = c[0], c[1]

            if output_groups and len(data) >= 4:
                g_data.append(data[3])
                g_type = g_data[-1].type()
            x_type = x.type()
            y_type = y.type()
            c_type = c.type()
            if not only_labels:
                x_data.append(x)
            y_data.append(y)
            c_data.append(c)
            if c_latent is not None:
                c_latent_data.append(c_latent)

        if not only_labels:
            x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)
        c_data = np.concatenate(c_data, axis=0)
        if g_data:
            g_data = np.concatenate(g_data, axis=0)
        else:
            g_data = np.ones((y_data.shape[0], 1), dtype=np.float32)
            g_type = torch.float32

        if c_latent_data:
            c_latent_data = np.concatenate(c_latent_data, axis=0)
        else:
            c_latent_data = None

        if as_torch:
            if not only_labels:
                x_data = torch.FloatTensor(x_data).type(x_type)
            y_data = torch.FloatTensor(y_data).type(y_type)
            c_data = torch.FloatTensor(c_data).type(c_type)
            if g_data is not None:
                g_data = torch.FloatTensor(g_data).type(g_type)
            if c_latent_data is not None:
                c_latent_data = torch.FloatTensor(c_latent_data).type(c_type)

    if output_latent_concepts:
        c_data = (c_data, c_latent_data)

    if output_groups:
        if not only_labels:
            return  x_data, y_data, c_data, g_data
        return  y_data, c_data, g_data
    if not only_labels:
        return  x_data, y_data, c_data
    return  y_data, c_data