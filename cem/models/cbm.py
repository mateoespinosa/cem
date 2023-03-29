import sklearn.metrics
import torch
import pytorch_lightning as pl
from torchvision.models import resnet50, densenet121
import numpy as np
import cem.train.utils as utils


################################################################################
## HELPER FUNCTIONS
################################################################################

def compute_bin_accuracy(c_pred, y_pred, c_true, y_true):
    c_pred = c_pred.reshape(-1).cpu().detach() > 0.5
    y_probs = y_pred.cpu().detach()
    y_pred = y_probs > 0.5
    c_true = c_true.reshape(-1).cpu().detach()
    y_true = y_true.reshape(-1).cpu().detach()
    c_accuracy = sklearn.metrics.accuracy_score(c_true, c_pred)
    c_auc = sklearn.metrics.roc_auc_score(c_true, c_pred, multi_class='ovo')
    c_f1 = sklearn.metrics.f1_score(c_true, c_pred, average='macro')
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    y_auc = sklearn.metrics.roc_auc_score(y_true, y_probs)
    y_f1 = sklearn.metrics.f1_score(y_true, y_pred)
    return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)


def compute_accuracy(
    c_pred,
    y_pred,
    c_true,
    y_true,
):
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
        return compute_bin_accuracy(
            c_pred,
            y_pred,
            c_true,
            y_true,
        )
    c_pred = (c_pred.cpu().detach().numpy() >= 0.5).astype(np.int32)
    # Doing the following transformation for when labels are not
    # fully certain
    c_true = (c_true.cpu().detach().numpy() > 0.5).astype(np.int32)
    y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
#     used_classes = np.unique(y_true.cpu().detach())
#     y_probs = y_probs[:, sorted(list(used_classes))]
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    y_true = y_true.cpu().detach()

    c_accuracy = c_auc = c_f1 = 0
    for i in range(c_true.shape[-1]):
        true_vars = c_true[:, i]
        pred_vars = c_pred[:, i]
        c_accuracy += sklearn.metrics.accuracy_score(
            true_vars, pred_vars
        ) / c_true.shape[-1]

        if len(np.unique(true_vars)) == 1:
            c_auc += np.mean(true_vars == pred_vars)/c_true.shape[-1]
        else:
            c_auc += sklearn.metrics.roc_auc_score(
                true_vars,
                pred_vars,
            )/c_true.shape[-1]
        c_f1 += sklearn.metrics.f1_score(
            true_vars,
            pred_vars,
            average='macro',
        )/c_true.shape[-1]
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class='ovo',
        )
    except Exception as e:
        y_auc = 0.0
    try:
        y_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    except:
        y_f1 = 0.0
    return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)


################################################################################
## BASELINE MODEL
################################################################################


class ConceptBottleneckModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=0.01,
        task_loss_weight=1,

        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        sigmoidal_extra_capacity=True,
        bottleneck_nonlinear=None,
        output_latent=False,

        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,

        top_k_accuracy=None,
        gpu=int(torch.cuda.is_available()),
    ):
        """
        Constructs a joint Concept Bottleneck Model (CBM) as defined by
        Koh et al. 2020.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CBM.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss. Default
            is 1.

        :param int extra_dims: The number of extra unsupervised dimensions to
            include in the bottleneck. Defaults to 0.
        :param Bool bool: Whether or not we threshold concepts in the bottleneck
            to be binary. Only relevant if the bottleneck uses a sigmoidal
            activation. Defaults to False.
        :param Bool sigmoidal_prob: Whether or not to use a sigmoidal activation
            for the bottleneck's activations that are aligned with training
            concepts. Defaults to True.
        :param Bool sigmoidal_extra_capacity:  Whether or not to use a sigmoidal
            activation for the bottleneck's unsupervised activations (when
            extra_dims > 0). Defaults to True.
        :param str bottleneck_nonlinear: A valid nonlinearity name to use for any
            unsupervised extra capacity in this model (when extra_dims > 0). It may
            overwrite `sigmoidal_extra_capacity` if sigmoidal_extra_capacity is
            True. If None, then no activation will be used. Will be soon deprecated.
            It must be one of [None, "sigmoid", "relu", "leakyrelu"] and defaults
            to None.

        :param Pytorch.Module x2c_model: A valid pytorch Module used to map the
            CBM's inputs to its bottleneck layer with `n_concepts + extra_dims`
            activations. If not given, then one may provide a generator
            function via the c_extractor_arch argument.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: If x2c_model is None,
            then one may provide a generator function for the input to concept
            model that takes as an input the size of the bottleneck (using
            an argument called `output_dim`) and returns a valid Pytorch Module
            that maps this CBM's inputs to the bottleneck of the requested size.
        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CBM's bottleneck (with size n_concepts + extra_dims`) to `n_tasks`
            output activations (i.e., the output of the CBM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.


        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization. Default is
            0.01.
        :param float weight_decay: The weight decay factor used during optimization.
            Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with n_tasks
            elements indicating the weights assigned to each output class
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.


        :param List[float] active_intervention_values: A list of n_concepts values
            to use when positively intervening in a given concept (i.e., setting
            concept c_i to 1 would imply setting its corresponding
            predicted concept to active_intervention_values[i]). If not given, then
            we will assume that we use `1` for all concepts. This parameter is
            important when intervening in CBMs that do not have sigmoidal concepts,
            as the intervention thresholds must then be inferred from their
            empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts values
            to use when negatively intervening in a given concept (i.e., setting
            concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts. This parameter is
            important when intervening in CBMs that do not have sigmoidal concepts,
            as the intervention thresholds must then be inferred from their
            empirical training distribution.
        :param Callable[(np.ndarray, np.ndarray, np.ndarray), np.ndarray] intervention_policy:
            An optional intervention policy to be used when intervening on a test
            batch sample x (first argument), with corresponding true concepts c
            (second argument), and true labels y (third argument). The policy must
            produce as an output a list of concept indices to intervene (in batch
            form) or a batch of binary masks indicating which concepts we will
            intervene on.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        :param Bool gpu: whether or not to use a GPU device or not.
        """
        gpu = int(gpu)
        super().__init__()
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        if x2c_model is not None:
            # Then this is assumed to be a module already provided as
            # the input to concepts method
            self.x2c_model = x2c_model
        else:
            self.x2c_model = c_extractor_arch(
                output_dim=(n_concepts + extra_dims)
            )

        # Now construct the label prediction model
        if c2y_model is not None:
            # Then this method has been provided to us already
            self.c2y_model = c2y_model
        else:
            # Else we construct it here directly
            units = [n_concepts + extra_dims] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)

        # Intervention-specific fields/handlers:
        init_fun = torch.cuda.FloatTensor if gpu else torch.FloatTensor
        if active_intervention_values is not None:
            self.active_intervention_values = init_fun(
                active_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.active_intervention_values = init_fun(
                [1 for _ in range(n_concepts)]
            ) * (
                5.0 if not sigmoidal_prob else 1.0
            )
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = init_fun(
                inactive_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.inactive_intervention_values = init_fun(
                [1 for _ in range(n_concepts)]
            ) * (
                -5.0 if not sigmoidal_prob else 0.0
            )

        # For legacy purposes, we wrap the model around a torch.nn.Sequential
        # module
        self.sig = torch.nn.Sigmoid()
        if sigmoidal_extra_capacity:
            # Keeping this for backwards compatability
            bottleneck_nonlinear = "sigmoid"
        if bottleneck_nonlinear == "sigmoid":
            self.bottleneck_nonlin = torch.nn.Sigmoid()
        elif bottleneck_nonlinear == "leakyrelu":
            self.bottleneck_nonlin = torch.nn.LeakyReLU()
        elif bottleneck_nonlinear == "relu":
            self.bottleneck_nonlin = torch.nn.ReLU()
        elif (bottleneck_nonlinear is None) or (
            bottleneck_nonlinear == "identity"
        ):
            self.bottleneck_nonlin = lambda x: x
        else:
            raise ValueError(
                f"Unsupported nonlinearity '{bottleneck_nonlinear}'"
            )

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.bool = bool
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.extra_dims = extra_dims
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_extra_capacity = sigmoidal_extra_capacity

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            y, c = batch[1]
        else:
            y, c = batch[1], batch[2]
        return x, y, c

    def _standardize_indices(self, intervention_idxs, batch_size):
        if isinstance(intervention_idxs, list):
            intervention_idxs = np.array(intervention_idxs)
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.IntTensor(intervention_idxs)

        if intervention_idxs is None or (
            isinstance(intervention_idxs, torch.Tensor) and
            ((len(intervention_idxs) == 0) or intervention_idxs.shape[-1] == 0)
        ):
            return None
        if not isinstance(intervention_idxs, torch.Tensor):
            raise ValueError(
                f'Unsupported intervention indices {intervention_idxs}'
            )
        if len(intervention_idxs.shape) == 1:
            # Then we will assume that we will do use the same
            # intervention indices for the entire batch!
            intervention_idxs = torch.tile(
                torch.unsqueeze(intervention_idxs, 0),
                (batch_size, 1),
            )
        elif len(intervention_idxs.shape) == 2:
            assert intervention_idxs.shape[0] == batch_size, (
                f'Expected intervention indices to have batch size {batch_size} '
                f'but got intervention indices with shape {intervention_idxs.shape}.'
            )
        else:
            raise ValueError(
                f'Intervention indices should have 1 or 2 dimensions. Instead we got '
                f'indices with shape {intervention_idxs.shape}.'
            )
        if intervention_idxs.shape[-1] == self.n_concepts:
            # We still need to check the corner case here where all indices are
            # given...
            elems = torch.unique(intervention_idxs)
            if len(elems) == 1:
                is_binary = (0 in elems) or (1 in elems)
            elif len(elems) == 2:
                is_binary = (0 in elems) and (1 in elems)
            else:
                is_binary = False
        else:
            is_binary = False
        if not is_binary:
            # Then this is an array of indices rather than a binary array!
            intervention_idxs = intervention_idxs.to(dtype=torch.long)
            result = torch.zeros(
                (batch_size, self.n_concepts),
                dtype=torch.bool,
                device=intervention_idxs.device,
            )
            result[:, intervention_idxs] = 1
            intervention_idxs = result
        assert intervention_idxs.shape[-1] == self.n_concepts, (
                f'Unsupported intervention indices with shape {intervention_idxs.shape}.'
            )
        if isinstance(intervention_idxs, np.ndarray):
            # Time to make it into a torch Tensor!
            intervention_idxs = torch.BoolTensor(intervention_idxs)
        intervention_idxs = intervention_idxs.to(dtype=torch.bool)
        return intervention_idxs

    def _concept_intervention(
        self,
        c_pred,
        intervention_idxs=None,
        c_true=None,
    ):
        if (c_true is None) or (intervention_idxs is None):
            return c_pred
        c_pred_copy = c_pred.clone()
        intervention_idxs = self._standardize_indices(
            intervention_idxs=intervention_idxs,
            batch_size=c_pred.shape[0],
        )
        intervention_idxs = intervention_idxs.to(c_pred.device)
        if self.sigmoidal_prob:
            c_pred_copy[intervention_idxs] = c_true[intervention_idxs]
        else:
            batched_active_intervention_values =  torch.tile(
                torch.unsqueeze(self.active_intervention_values, 0),
                (c_pred.shape[0], 1),
            )

            batched_inactive_intervention_values =  torch.tile(
                torch.unsqueeze(self.inactive_intervention_values, 0),
                (c_pred.shape[0], 1),
            )

            c_pred_copy[intervention_idxs] = (
                (
                    c_true[intervention_idxs] *
                    batched_active_intervention_values[intervention_idxs]
                ) +
                (
                    (c_true[intervention_idxs] - 1) *
                    -batched_inactive_intervention_values[intervention_idxs]
                )
            )

        return c_pred_copy

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
    ):
        if latent is None:
            latent = self.x2c_model(x)
        if self.sigmoidal_prob or self.bool:
            if self.extra_dims:
                # Then we only sigmoid on the probability bits but
                # let the other entries up for grabs
                c_pred_probs = self.sig(latent[:, :-self.extra_dims])
                c_others = self.bottleneck_nonlin(latent[:,-self.extra_dims:])
                c_pred =  torch.cat([c_pred_probs, c_others], axis=-1)
            else:
                c_pred = self.sig(latent)
            # And the semantics vector is just the predictions
            if self.extra_dims:
                c_sem = latent[:, :-self.extra_dims]
            else:
                c_sem = latent
        else:
            # Otherwise, the concept vector itself is not sigmoided
            # but the semantics
            if self.extra_dims:
                c_sem = self.sig(latent[:, :-self.extra_dims])
            else:
                c_sem = self.sig(latent)
        # Now include any interventions that we may want to include
        if (intervention_idxs is None) and (
            self.intervention_policy is not None
        ):
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
            )
        else:
            c_int = c
        c_pred = self._concept_intervention(
            c_pred=latent,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )
        if self.bool:
            y = self.c2y_model((c_pred > 0.5).float())
        else:
            y = self.c2y_model(c_pred)
        if self.output_latent:
            return c_sem, c_pred, y, latent
        return c_sem, c_pred, y

    def forward(self, x, c=None, y=None, latent=None, intervention_idxs=None):
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            intervention_idxs=intervention_idxs,
            latent=latent,
        )

    def predict_step(
        self,
        batch,
        batch_idx,
        intervention_idxs=None,
        dataloader_idx=0,
    ):
        x, y, c = self._unpack_batch(batch)
        return self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=False,
        )

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, c = self._unpack_batch(batch)
        if self.output_latent:
            c_sem, c_logits, y_logits, _ = self._forward(
                x,
                intervention_idxs=intervention_idxs,
                c=c,
                y=y,
                train=train,
            )
        else:
            c_sem, c_logits, y_logits = self._forward(
                x,
                intervention_idxs=intervention_idxs,
                c=c,
                y=y,
                train=train,
            )
        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0
            task_loss_scalar = 0
        if self.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            concept_loss = self.loss_concept(c_sem, c)
            loss = self.concept_loss_weight * concept_loss + task_loss
            concept_loss_scalar = concept_loss.detach()
        else:
            loss = task_loss
            concept_loss_scalar = 0.0
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                    y_true,
                    y_pred,
                    k=top_k_val,
                    labels=labels,
                )
                result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            self.log(name, val, prog_bar=("accuracy" in name))
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result['c_accuracy'],
                "c_auc": result['c_auc'],
                "c_f1": result['c_f1'],
                "y_accuracy": result['y_accuracy'],
                "y_auc": result['y_auc'],
                "y_f1": result['y_f1'],
                "concept_loss": result['concept_loss'],
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "avg_c_y_acc": result['avg_c_y_acc'],
            },
        }

    def validation_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("val_" + name, val, prog_bar=("accuracy" in name))
        return {
            "val_" + key: val
            for key, val in result.items()
        }

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result['loss']

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
