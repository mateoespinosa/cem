import argparse
import copy
import joblib
import json
import logging
import numpy as np
import os
import sys
import torch
import yaml
import pytorch_lightning as pl


from datetime import datetime
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, RandomSampler

import cem.data.celeba_loader as celeba_data_module
import cem.data.mnist_add as mnist_data_module
from cem.models.acflow import ACFlow

################################################################################
## MAIN FUNCTION
################################################################################

# Helper class to apply transformations to a dataset
class TransformedDataset(Dataset):
    def __init__(self, dataset, n_tasks):
        self.dataset = dataset
        self.n_tasks = n_tasks

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        if len(batch) > 3:
            competencies = batch[3]
        else:
            competencies = None
        if len(batch) > 4:
            prev_interventions = batch[4]
        else:
            prev_interventions = None
        return x, y, (c, competencies, prev_interventions)
    
    def transform(self, batch):
        _, y, (x, _, _) = self._unpack_batch(batch)
        d = x.shape[-1]
        b = np.zeros([d], dtype=np.float32)
        no = np.random.choice(d+1)
        o = np.random.choice(d, [no], replace=False)
        b[o] = 1.
        m = b.copy()
        w = list(np.where(b == 0)[0])
        w.append(-1)
        w = np.random.choice(w)
        if w >= 0:
            m[w] = 1.
        b = torch.tensor(b)
        m = torch.tensor(m)
        b.to(x.device)
        m.to(x.device)
        if self.n_tasks > 1:
            y = torch.nn.functional.one_hot(
                y,
                num_classes=self.n_tasks,
            )
        else:
            y = torch.cat(
                [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                dim=-1,
            )
        return {'x': x, 'b': b, 'm': m, 'y': y}

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

def transform_dataloader(dataloader, n_tasks):
    dataset = TransformedDataset(dataloader.dataset, n_tasks)
    return torch.utils.data.DataLoader(dataset, batch_size = dataloader.batch_size, shuffle = isinstance(dataloader.sampler, RandomSampler), num_workers = dataloader.num_workers)



def main(
    data_module,
    experiment_config,
    num_workers=8,
    accelerator="auto",
    devices="auto"
):
    seed_everything(42)
    # parameters for data, model, and training
    experiment_config = copy.deepcopy(experiment_config)
    if 'shared_params' not in experiment_config:
        experiment_config['shared_params'] = {}
    # Move all global things into the shared params
    for key, vals in experiment_config.items():
        if key not in ['runs', 'shared_params']:
            experiment_config['shared_params'][key] = vals
    experiment_config['shared_params']['num_workers'] = num_workers
    logging.debug(
        f"Processing dataset..."
    )
    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
        data_module.generate_data(
            config=experiment_config['shared_params'],
            seed=42,
            output_dataset_vars=True,
            root_dir=experiment_config['shared_params'].get('root_dir', None),
        )
    logging.debug(
        f"Applying transformations..."
    )
    train_dl = transform_dataloader(train_dl, n_tasks)
    # For now, we assume that all concepts have the same
    # aquisition cost
    experiment_config["shared_params"]["n_concepts"] = \
        experiment_config["shared_params"].get(
            "n_concepts",
            n_concepts,
        )
    experiment_config["shared_params"]["n_tasks"] = \
        experiment_config["shared_params"].get(
            "n_tasks",
            n_tasks,
        )

    
    logging.info(
        f"\tNumber of output classes: {n_tasks}"
    )
    logging.info(
        f"\tNumber of training concepts: {n_concepts}"
    )
    val_dl = transform_dataloader(val_dl, n_tasks)
    test_dl = transform_dataloader(test_dl, n_tasks)

    sample = next(iter(train_dl.dataset))

    logging.debug(
        f"Sample: {sample}"
    )

    task_class_weights = None

    if experiment_config['shared_params'].get('use_task_class_weights', False):
        logging.info(
            f"Computing task class weights in the training dataset with "
            f"size {len(train_dl)}..."
        )
        attribute_count = np.zeros((max(n_tasks, 2),))
        samples_seen = 0
        for i, data in enumerate(train_dl):
            y = data['y']
            import pdb
            pdb.set_trace()
            attribute_count += np.sum(np.array(y), axis=0)
            samples_seen += y.shape[0]
        print("Class distribution is:", attribute_count / samples_seen)
        if n_tasks > 1:
            task_class_weights = samples_seen / attribute_count - 1
        else:
            task_class_weights = np.array(
                [attribute_count[0]/attribute_count[1]]
            )


    # Set log level in env variable as this will be necessary for
    # subprocessing
    os.environ['LOGLEVEL'] = os.environ.get(
        'LOGLEVEL',
        logging.getLevelName(logging.getLogger().getEffectiveLevel()),
    )
    loglevel = os.environ['LOGLEVEL']
    logging.info(f'Setting log level to: "{loglevel}"')

    results = {}
    for split in range(
        experiment_config['shared_params'].get("start_split", 0),
        experiment_config['shared_params']["trials"],
    ):
        results[f'{split}'] = {}
        now = datetime.now()
        print(
            f"[TRIAL "
            f"{split + 1}/{experiment_config['shared_params']['trials']} "
            f"BEGINS AT {now.strftime('%d/%m/%Y at %H:%M:%S')}"
        )
        # And then over all runs in a given trial
        
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=experiment_config['shared_params'].get('max_epochs', 500),
            logger=False,
        )

        model = ACFlow(
            n_concepts = n_concepts, 
            n_tasks = n_tasks,
            layer_cfg = experiment_config['shared_params']['layer_cfg'], 
            affine_hids = experiment_config['shared_params']['affine_hids'], 
            linear_rank = experiment_config['shared_params']['linear_rank'],
            linear_hids = experiment_config['shared_params']['linear_hids'], 
            transformations = experiment_config['shared_params']['transform'], 
            optimizer = experiment_config['shared_params']['optimizer'], 
            learning_rate = experiment_config['shared_params']['learning_rate'], 
            weight_decay = experiment_config['shared_params']['decay_rate'], 
            momentum = experiment_config['shared_params'].get('momentum', 0.9), 
            prior_units = experiment_config['shared_params']['prior_units'], 
            prior_layers = experiment_config['shared_params']['prior_layers'], 
            prior_hids = experiment_config['shared_params']['prior_hids'], 
            n_components = experiment_config['shared_params']['n_components'], 
            lambda_xent = 1, 
            lambda_nll = 1
        )

        logging.debug(
            f"Starting model training..."
        )

        trainer.fit(model, train_dl, val_dl)
        model.freeze()

        logging.debug(
            f"Starting model training..."
        )

        [test_results] = trainer.test(model, test_dl)

        acc = test_results['test_acc']
        nll = test_results['test_nll']
        logging.debug(
            f"\tTest Accuracy is {acc}\n"
            f"\tNLL is {nll}\n"
        )

    return results


################################################################################
## Arg Parser
################################################################################


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Runs AC Flow experiments in a given dataset.'
        ),
    )
    parser.add_argument(
        '--config',
        '-c',
        default=None,
        help=(
            "YAML file with the configuration for the experiment. If not "
            "provided, then we will use the default configuration for the "
            "dataset."
        ),
        metavar="config.yaml",
    )
    parser.add_argument(
        '--dataset',
        choices=[
            'cub',
            'celeba',
            'mnist_add'
        ],
        help=(
            "Dataset to run experiments for. Must be a supported dataset with "
            "a loader."
        ),
        metavar="ds_name",
        default=None,
    )
    parser.add_argument(
        '--num_workers',
        default=8,
        help=(
            'number of workers used for data feeders. Do not use more workers '
            'than cores in the machine.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        default=False,
        help="forces CPU training.",
    )
    return parser


################################################################################
## Main Entry Point
################################################################################

if __name__ == '__main__':
    # Build our arg parser first
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    if args.config:
        with open(args.config, "r") as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        loaded_config = {}

    if args.dataset is not None:
        loaded_config["dataset"] = args.dataset
    if loaded_config.get("dataset", None) is None:
        raise ValueError(
            "A dataset must be provided either as part of the "
            "configuration file or as a command line argument."
        )
    # if loaded_config["dataset"] == "cub":
    #     data_module = cub_data_module
    if loaded_config["dataset"] == "celeba":
        data_module = celeba_data_module
    elif loaded_config["dataset"] == "mnist_add":
        data_module = mnist_data_module
        num_operands = loaded_config.get('num_operands', 32)
    else:
        raise ValueError(f"Unsupported dataset {loaded_config['dataset']}!")

    main(
        data_module=data_module,
        accelerator=(
            "gpu" if (not args.force_cpu) and (torch.cuda.is_available())
            else "cpu"
        ),
        experiment_config=loaded_config
    )
