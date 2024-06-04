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
):
    if hasattr(dl.dataset, 'tensors'):
        x_data, y_data, c_data = dl.dataset.tensors[:3]
        if output_groups:
            if len(dl.dataset.tensors) >= 4:
                g_data = dl.dataset.tensors[3]
            else:
                g_data =  np.ones((x_data.shape[0], 1), dtype=np.int32)
        if not as_torch:
            x_data = x_data.detach().cpu().numpy()
            y_data = y_data.detach().cpu().numpy()
            c_data = c_data.detach().cpu().numpy()
            if output_groups and (not isinstance(g_data, np.ndarray)):
                g_data = g_data.detach().cpu().numpy()
    else:
        fast_loader = torch.utils.data.DataLoader(
            dl.dataset,
            batch_size=_largest_divisor(len(dl.dataset), max_val=max_val),
            num_workers=num_workers,
        )
        x_data, y_data, c_data, g_data = [], [], [], []
        for data in fast_loader:
            if len(data) == 2:
                x, (y, c) = data
            else:
                (x, y, c) = data[:3]
            if output_groups and len(data) >= 4:
                g_data.append(data[3])
                g_type = g_data[-1].type()
            x_type = x.type()
            y_type = y.type()
            c_type = c.type()
            x_data.append(x)
            y_data.append(y)
            c_data.append(c)

        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)
        c_data = np.concatenate(c_data, axis=0)
        if g_data:
            g_data = np.concatenate(g_data, axis=0)
        else:
            g_data = np.ones((x_data.shape[0], 1), dtype=np.int32)
            g_type = torch.int32

        if as_torch:
            x_data = torch.FloatTensor(x_data).type(x_type)
            y_data = torch.FloatTensor(y_data).type(y_type)
            c_data = torch.FloatTensor(c_data).type(c_type)
            if g_data is not None:
                g_data = torch.FloatTensor(g_data).type(g_type)
    if output_groups:
        return  x_data, y_data, c_data, g_data
    return  x_data, y_data, c_data