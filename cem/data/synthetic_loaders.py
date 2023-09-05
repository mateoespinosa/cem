import numpy as np
import torch

from pytorch_lightning import seed_everything

def generate_xor_data(size):
    # sample from normal distribution
    x = np.random.uniform(0, 1, (size, 2))
    c = np.stack([
        x[:, 0] > 0.5,
        x[:, 1] > 0.5,
    ]).T
    y = np.logical_xor(c[:, 0], c[:, 1])

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)
    return x, c, y


def generate_trig_data(size):
    h = np.random.normal(0, 2, (size, 3))
    x, y, z = h[:, 0], h[:, 1], h[:, 2]

    # raw features
    input_features = np.stack([
        np.sin(x) + x,
        np.cos(x) + x,
        np.sin(y) + y,
        np.cos(y) + y,
        np.sin(z) + z,
        np.cos(z) + z,
        x ** 2 + y ** 2 + z ** 2,
    ]).T

    # concepts
    concepts = np.stack([
        x > 0,
        y > 0,
        z > 0,
    ]).T

    # task
    downstream_task = (x + y + z) > 1

    input_features = torch.FloatTensor(input_features)
    concepts = torch.FloatTensor(concepts)
    downstream_task = torch.FloatTensor(downstream_task)
    return input_features, concepts, downstream_task


def generate_dot_data(size):
    # sample from normal distribution
    emb_size = 2
    v1 = np.random.randn(size, emb_size) * 2
    v2 = np.ones(emb_size)
    v3 = np.random.randn(size, emb_size) * 2
    v4 = -np.ones(emb_size)
    x = np.hstack([v1+v3, v1-v3])
    c = np.stack([
        np.dot(v1, v2).ravel() > 0,
        np.dot(v3, v4).ravel() > 0,
    ]).T
    y = ((v1*v3).sum(axis=-1) > 0).astype(np.int64)

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.Tensor(y)
    return x, c, y

class SyntheticGenerator(object):
    def __init__(self, dataset_name):
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower == "xor":
            generate_data = generate_xor_data
            n_tasks = 1
            num_concepts = 2

        elif dataset_name_lower in ["trig", "trigonometry"]:
            generate_data = generate_trig_data
            n_tasks = 1
            num_concepts = 3

        elif dataset_name_lower in ["vector", "dot"]:
            generate_data = generate_dot_data
            num_concepts = 2
            n_tasks = 1
        else:
            raise ValueError(f"Unsupported dataset name {dataset_name}")

        def _data_loader(
            config,
            root_dir=None,
            seed=42,
            output_dataset_vars=False,
        ):
            seed_everything(seed)

            dataset_size = config.get('dataset_size', 3000)
            batch_size = config["batch_size"]
            x, c, y = generate_data(int(dataset_size * 0.7))
            train_data = torch.utils.data.TensorDataset(x, y, c)
            train_dl = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
            )

            x_test, c_test, y_test = generate_data(int(dataset_size * 0.2))
            test_data = torch.utils.data.TensorDataset(x_test, y_test, c_test)
            test_dl = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
            )

            x_val, c_val, y_val = generate_data(int(dataset_size * 0.1))
            val_data = torch.utils.data.TensorDataset(x_val, y_val, c_val)
            val_dl = torch.utils.data.DataLoader(
                val_data,
                batch_size=batch_size,
            )

            if config.get('weight_loss', False):
                attribute_count = np.zeros((num_concepts,))
                samples_seen = 0
                for i, (_, y, c) in enumerate(train_dl):
                    c = c.cpu().detach().numpy()
                    attribute_count += np.sum(c, axis=0)
                    samples_seen += c.shape[0]
                imbalance = samples_seen / attribute_count - 1
            else:
                imbalance = None
            if not output_dataset_vars:
                return train_dl, val_dl, test_dl, imbalance
            concept_group_map = dict([(i, [i]) for i in range(num_concepts)])
            return (
                train_dl,
                val_dl,
                test_dl,
                imbalance,
                (num_concepts, n_tasks, concept_group_map),
            )
        self.generate_data = _data_loader

def get_synthetic_num_features(dataset_name):
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "xor":
        return 2
    elif dataset_name_lower in ["trig", "trigonometry"]:
        return 7
    elif dataset_name_lower in ["vector", "dot"]:
        return 4
    raise ValueError(f"Unsupported dataset name {dataset_name}")

def get_synthetic_data_loader(dataset_name):
    return SyntheticGenerator(dataset_name)