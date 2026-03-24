import sklearn.metrics
import torch
import pytorch_lightning as pl
from torchvision.models import resnet50
import numpy as np

import cem.train.utils as utils
from cem.metrics.accs import compute_accuracy

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
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        # New additions
        training_intervention_prob=0.0,
        prior_loss_term=0.0,
        prior_only_concepts=False,

        top_k_accuracy=None,
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
            component corresponding to the output task classification loss.
            Default is 1.

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
        :param str bottleneck_nonlinear: A valid nonlinearity name to use for
            any unsupervised extra capacity in this model (when extra_dims > 0).
            It may overwrite `sigmoidal_extra_capacity` if
            sigmoidal_extra_capacity is True. If None, then no activation will
            be used. Will be soon deprecated. It must be one of [None,
            "sigmoid", "relu", "leakyrelu"] and defaults to None.

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
            use a simple linear layer to map the bottleneck to the output
            classes.


        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.


        :param List[float] active_intervention_values: A list of n_concepts
            values to use when positively intervening in a given concept (i.e.,
            setting concept c_i to 1 would imply setting its corresponding
            predicted concept to active_intervention_values[i]). If not given,
            then we will assume that we use `1` for all concepts. This
            parameter is important when intervening in CBMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts. This
            parameter is important when intervening in CBMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param Callable[(np.ndarray, np.ndarray, np.ndarray), np.ndarray] intervention_policy:
            An optional intervention policy to be used when intervening on a
            test batch sample x (first argument), with corresponding true
            concepts c (second argument), and true labels y (third argument).
            The policy must produce as an output a list of concept indices to
            intervene (in batch form) or a batch of binary masks indicating
            which concepts we will intervene on.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        """
        super().__init__()
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.output_interventions = output_interventions
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
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
        if active_intervention_values is not None:
            self.active_intervention_values = torch.FloatTensor(
                active_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.active_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * (
                5.0 if not sigmoidal_prob else 1.0
            )
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.FloatTensor(
                inactive_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.inactive_intervention_values = torch.FloatTensor(
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
                pos_weight=task_class_weights
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
        self.use_concept_groups = use_concept_groups
        self.training_intervention_prob = training_intervention_prob
        self.prior_loss_term = prior_loss_term
        self.prior_only_concepts = prior_only_concepts

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            offset = 2
            y, c = batch[1]
        else:
            offset = 3
            y, c = batch[1], batch[2]
        if len(batch) > (offset):
            g = batch[offset]
        else:
            g = None
        if len(batch) > (offset + 1):
            competencies = batch[offset + 1]
        else:
            competencies = None
        if len(batch) > (offset + 2):
            prev_interventions = batch[offset + 2]
        else:
            prev_interventions = None
        if len(batch) > (offset + 3):
            # Then we are given latent concepts as an input too. Let's
            # pack those up with the known training concepts
            c = (c, batch[offset + 3])
        return x, y, (c, g, competencies, prev_interventions)

    def _standardize_indices(self, intervention_idxs, batch_size, device='cuda'):
        if getattr(self, 'force_all_interventions', False):
            intervention_idxs = torch.ones(
                (batch_size, self.n_concepts)
            ).to(device)
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
                f'but got intervention indices with '
                f'shape {intervention_idxs.shape}.'
            )
        else:
            raise ValueError(
                f'Intervention indices should have 1 or 2 dimensions. Instead '
                f'we got indices with shape {intervention_idxs.shape}.'
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
                f'Unsupported intervention indices with '
                f'shape {intervention_idxs.shape}.'
            )
        if isinstance(intervention_idxs, np.ndarray):
            # Time to make it into a torch Tensor!
            intervention_idxs = torch.BoolTensor(intervention_idxs)
        intervention_idxs = intervention_idxs.to(dtype=torch.bool)
        return intervention_idxs

    def _extra_losses(
        self,
        x,
        y,
        c,
        y_pred,
        c_sem,
        c_pred,
        competencies=None,
        prev_interventions=None,
    ):
        loss = 0.0
        if getattr(self, 'prior_loss_term', 0.0) > 0.0:
            # Then construct a bottleneck where all concepts are intervened
            bottleneck_gt = self._concept_intervention(
                c_pred=c_pred,
                intervention_idxs=torch.ones(
                    (c_pred.shape[0], self.n_concepts)
                ).to(c_pred.device),
                c_true=c,
            )
            if getattr(self, 'prior_only_concepts', False) and (
                getattr(self, 'extra_dims', 0) > 0
            ):
                # Then we will block all information from the extra capacity
                # that is not concept specific.
                bottleneck_gt[:, self.n_concepts:] = 0.0
            # Make a label prediction based on this fully intervened bottleneck
            y_pred_gt = self.c2y_model(
                bottleneck_gt if not self.bool else
                (bottleneck_gt > 0.5).float()
            )
            loss += self.prior_loss_term * self.loss_task(
                y_pred_gt,
                y,
            )
        return loss

    def _prior_int_distribution(
        self,
        c,
        prob,
        pos_embeddings,
        neg_embeddings,
        competencies=None,
        prev_interventions=None,
        train=False,
        horizon=1,
    ):
        return None

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
            device=c_pred.device,
        )
        intervention_idxs = intervention_idxs.to(c_pred.device)
        # Check whether the mask needs to be extended because of
        # extra dimensions
        if self.extra_dims:
            set_intervention_idxs = torch.nn.functional.pad(
                intervention_idxs,
                pad=(0, self.extra_dims),  # Just pads the last dimension
            )
        else:
            set_intervention_idxs = intervention_idxs
        if self.sigmoidal_prob:
            c_pred_copy[set_intervention_idxs] = c_true[intervention_idxs]
        else:
            active_intervention_values = self.active_intervention_values.to(
                c_pred.device
            )
            batched_active_intervention_values =  torch.tile(
                torch.unsqueeze(active_intervention_values, 0),
                (c_pred.shape[0], 1),
            ).to(c_true.device)

            inactive_intervention_values = self.inactive_intervention_values.to(
                c_pred.device
            )
            batched_inactive_intervention_values = torch.tile(
                torch.unsqueeze(inactive_intervention_values, 0),
                (c_pred.shape[0], 1),
            ).to(c_true.device)

            c_pred_copy[set_intervention_idxs] = (
                (
                    c_true[intervention_idxs] *
                    batched_active_intervention_values[intervention_idxs]
                ) +
                (
                    (1 - c_true[intervention_idxs]) *
                    batched_inactive_intervention_values[intervention_idxs]
                )
            )
        return c_pred_copy

    def _forward(
        self,
        x,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        output_latent=None,
        output_embeddings=False,
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            latent = self.x2c_model(x)
        if self.sigmoidal_prob or self.bool:
            if self.extra_dims:
                # Then we only sigmoid on the probability bits but
                # let the other entries up for grabs
                c_pred_probs = self.sig(latent[:, :-self.extra_dims])
                c_others = self.bottleneck_nonlin(latent[:,-self.extra_dims:])
                c_pred =  torch.cat([c_pred_probs, c_others], axis=-1)
                c_sem = c_pred_probs
            else:
                c_pred = self.sig(latent)
                c_sem = c_pred
        else:
            # Otherwise, the concept vector itself is not sigmoided
            # but the semantics
            c_pred = latent
            if self.extra_dims:
                c_sem = self.sig(latent[:, :-self.extra_dims])
            else:
                c_sem = self.sig(latent)
        if output_embeddings or (
            (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        )):
            pos_embeddings = torch.ones(c_sem.shape).to(x.device)
            neg_embeddings = torch.zeros(c_sem.shape).to(x.device)
            if not (self.sigmoidal_prob or self.bool):
                if (
                    (self.active_intervention_values is not None) and
                    (self.inactive_intervention_values is not None)
                ):
                    active_intervention_values = \
                        self.active_intervention_values.to(
                            c_pred.device
                        )
                    pos_embeddings = torch.tile(
                        active_intervention_values,
                        (c.shape[0], 1),
                    ).to(active_intervention_values.device)
                    inactive_intervention_values = \
                        self.inactive_intervention_values.to(
                            c_pred.device
                        )
                    neg_embeddings = torch.tile(
                        inactive_intervention_values,
                        (c.shape[0], 1),
                    ).to(inactive_intervention_values.device)
                else:
                    out_embs = c_pred.detach().cpu().numpy()
                    for concept_idx in range(self.n_concepts):
                        pos_embeddings[:, concept_idx] = np.percentile(
                            out_embs[:, concept_idx],
                            95,
                        )
                        neg_embeddings[:, concept_idx] = np.percentile(
                            out_embs[:, concept_idx],
                            5,
                        )
            pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
            neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)
        # Now include any interventions that we may want to include
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            prior_distribution = self._prior_int_distribution(
                c=c,
                prob=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                competencies=competencies,
                prev_interventions=prev_interventions,
                train=train,
                horizon=1,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )
        else:
            c_int = c
        if train and (self.training_intervention_prob > 0.0) and (
            intervention_idxs is None
        ):
            intervention_idxs = torch.rand(
                (c_pred.shape[0], self.n_concepts)
            ).to(x.device) < self.training_intervention_prob

        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )
        if self.bool:
            y_pred = self.c2y_model((c_pred > 0.5).float())
        else:
            y_pred = self.c2y_model(c_pred)
        tail_results = []
        if output_interventions:
            if intervention_idxs is None:
                intervention_idxs = None
            if isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(pos_embeddings)
            tail_results.append(neg_embeddings)
        tail_results += self._extra_tail_results(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        return tuple([c_sem, c_pred, y_pred] + tail_results)

    def _extra_tail_results(
        self,
        x,
        y,
        c,
        c_sem,
        competencies,
        prev_interventions,
    ):
        return []

    def forward(
        self,
        x,
        c=None,
        y=None,
        latent=None,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        **kwargs
    ):
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            competencies=competencies,
            prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs,
            latent=latent,
            **kwargs
        )

    def predict_step(
        self,
        batch,
        batch_idx,
        intervention_idxs=None,
        dataloader_idx=0,
    ):
        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(
            batch
        )
        return self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=getattr(self, 'output_embeddings', False),
        )

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(
            batch
        )
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        if isinstance(c, (list, tuple)):
            # Then we were provided with a set of training concepts and a
            # set of latent concepts. Let's just use the training concepts
            # for evaluation
            c = c[0]
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
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
            # Will only compute the concept loss for concepts whose certainty
            # values are fully given
            concept_loss = self.loss_concept(c_sem, c)
            concept_loss_scalar = concept_loss.detach()
            loss = self.concept_loss_weight * concept_loss + task_loss + \
                self._extra_losses(
                    x=x,
                    y=y,
                    c=c,
                    c_sem=c_sem,
                    c_pred=c_logits,
                    y_pred=y_logits,
                    competencies=competencies,
                    prev_interventions=prev_interventions,
                )
        else:
            loss = task_loss + self._extra_losses(
                x=x,
                y=y,
                c=c,
                c_sem=c_sem,
                c_pred=c_logits,
                y_pred=y_logits,
                competencies=competencies,
                prev_interventions=prev_interventions,
            )
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
            if isinstance(self.top_k_accuracy, int):
                top_k_accuracy = list(range(1, self.top_k_accuracy))
            else:
                top_k_accuracy = self.top_k_accuracy

            for top_k_val in top_k_accuracy:
                if top_k_val:
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
            if self.n_tasks <= 2:
                prog_bar = (
                    ("auc" in name) or
                    ("mask_accuracy" in name) or
                    ("current_steps" in name) or
                    ("num_rollouts" in name) or
                    ("mode" in name) or
                    ("adv" in name)
                )
            else:
                prog_bar = (
                    ("c_auc" in name) or
                    ("y_accuracy" in name) or
                    ("mask_accuracy" in name) or
                    ("current_steps" in name) or
                    ("num_rollouts" in name) or
                    ("mode" in name) or
                    ("adv" in name)

                )
            self.log(name, val, prog_bar=prog_bar)
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result.get('c_accuracy', 0),
                "c_auc": result.get('c_auc', 0),
                "c_f1": result.get('c_f1', 0),
                "y_accuracy": result.get('y_accuracy', 0),
                "y_auc": result.get('y_auc', 0),
                "y_f1": result.get('y_f1', 0),
                "concept_loss": result.get('concept_loss', 0),
                "task_loss": result.get('task_loss', 0),
                "loss": result.get('loss', 0),
                "avg_c_y_acc": result.get('avg_c_y_acc', 0),
            },
        }

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = (("auc" in name))
            else:
                prog_bar = (("c_auc" in name) or ("y_accuracy" in name))
            self.log("val_" + name, val, prog_bar=prog_bar)
        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        return result

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
        if self.lr_scheduler_patience != 0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                patience=self.lr_scheduler_patience,
                factor=self.lr_scheduler_factor,
                min_lr=getattr(self, 'min_lr', 1e-5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "loss",
            }
        return {
            "optimizer": optimizer,
            "monitor": "loss",
        }


class EntangledHybridCBM(ConceptBottleneckModel):
    """A joint hybrid Concept Bottleneck Model (CBM) (a CBM with extra
    unsupervised capacity in its bottleneck) whose unsupervised capacity
    is a function of the concepts themselves (so they it is entangled with
    the concepts and is affected by concept interventions).
    """
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=0.01,
        task_loss_weight=1,

        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        bottleneck_nonlinear=None,
        output_latent=False,
        latent_dim=None,

        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        # New additions
        training_intervention_prob=0.0,
        prior_loss_term=0.0,
        prior_only_concepts=False,
        entanglement_mode='additive',
        l2_reg_residual_weights=0.0,

        top_k_accuracy=None,
    ):
        """Initializes a EntangledHybridCBM instance.

        :param int n_concepts: Number of concepts in the CBM's bottleneck.
        :param int n_tasks: Number of tasks (i.e., output classes) to predict.
        :param float concept_loss_weight: Weight assigned to the concept loss
            during training. Default is 0.01.
        :param float task_loss_weight: Weight assigned to the task loss during
            training. Default is 1.


        :param int extra_dims: Number of extra unsupervised dimensions to add
            to the CBM's bottleneck (i.e., the bottleneck will have size
            `n_concepts + extra_dims`). Default is 0.
        :param bool bool: Whether the concept predictions should be treated
            as binary values rather than probabilistic values. Default is False.
        :param bool sigmoidal_prob: Whether the concept predictions should be
            produced by applying a sigmoid non-linearity to the bottleneck
            activations. If False, then the concept predictions will be the
            raw bottleneck activations. Default is True.
        :param str bottleneck_nonlinear: The non-linearity to apply to the
            bottleneck activations. Must be one of `sigmoid`, `leakyrelu`,
            `relu`, `identity` or None. Default is None.
        :param bool output_latent: Whether to output the latent
            representations when doing a forward pass. Default is False.
        :param torch.nn.Module x2c_model: An optional model mapping inputs
            to concepts. If not given, then we will use c_extractor_arch to
            construct a concept extractor model.
        :param Callable[[], torch.nn.Module] c_extractor_arch: A callable
            that produces a concept extractor model when called. This is only
            used when x2c_model is not given. Default is a ResNet-50 based
            model.
        :param torch.nn.Module c2y_model: An optional model mapping concepts
            to task labels. If not given, then we will construct a
            fully-connected network based on c2y_layers.
        :param List[int] c2y_layers: Either None or a list indicating the
            number of units in each hidden layer of the concept-to-labels
            model. For example, [100, 50] would indicate a model with two
            hidden layers, with 100 units in the first layer and 50 units in
            the second layer. If None, then no hidden layers will be used.
        :param str optimizer: The name of the optimizer to use. Either
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: The momentum to use when using SGD. Default is
            0.9.
        :param float learning_rate: The learning rate to use during training.
            Default is 0.01.
        :param float weight_decay: The weight decay (L2 penalty) to use
            during training. Default is 4e-05.
        :param float lr_scheduler_factor: The factor by which to reduce
            the learning rate when using a learning rate scheduler. Default is
            0.1.
        :param int lr_scheduler_patience: The number of epochs with no
            improvement after which the learning rate will be reduced when
            using a learning rate scheduler. Default is 10. If set to 0,
            then no learning rate scheduler will be used.
        :param torch.Tensor weight_loss: An optional tensor of shape
            (n_concepts,) indicating the weight to assign to each concept
            during concept loss computation. If None, then all concepts
            will be equally weighted.
        :param torch.Tensor task_class_weights: An optional tensor of shape
            (n_tasks,) indicating the weight to assign to each task class
            during task loss computation. If None, then all classes will be
            equally weighted.
        :param List[float] active_intervention_values: An optional list of
            length n_concepts indicating the values to set intervened concepts
            to when performing an intervention to set a concept to be active
            (i.e., concept value 1). If None, then a default value of 5.0
            (or 1.0 if sigmoidal_prob is True) will be used for all concepts.
        :param List[float] inactive_intervention_values: An optional list of
            length n_concepts indicating the values to set intervened concepts
            to when performing an intervention to set a concept to be inactive
            (i.e., concept value 0). If None, then a default value of -5.0
            (or 0.0 if sigmoidal_prob is True) will be used for all concepts.
        :param intervention_policy: An optional intervention policy
            (an instance of cem.policies.InterventionPolicy) that will be used
            to select interventions during training and evaluation when
            intervention indices are not provided. If None, then no interventions
            will be performed unless intervention indices are provided during
            training/evaluation.
        :param bool output_interventions: Whether to output the intervention
            indices when doing a forward pass. Default is False.
        :param bool use_concept_groups: Whether to use concept groups
            (cem.models.concept_groups.ConceptGroups) when performing
            interventions. Default is False.
        :param float training_intervention_prob: The probability of randomly
            intervening on each concept during training. Default is 0.0.
        :param float prior_loss_term: The weight assigned to an auxiliary
            loss term that encourages the model to make correct predictions
            when all concepts are intervened upon. Default is 0.0.
        :param top_k_accuracy: Either None or an integer k indicating that
            top-k accuracy should be computed during evaluation. If a list
            of integers is given, then top-k accuracy will be computed for
            each k in the list.
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.output_interventions = output_interventions
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.latent_code_generator = None
        latent_dim = latent_dim or n_concepts
        if x2c_model is not None:
            # Then this is assumed to be a module already provided as
            # the input to concepts method
            self.latent_code_generator = x2c_model
        else:
            self.latent_code_generator = c_extractor_arch(
                output_dim=(latent_dim)
            )
        # The x2c model will take the latent code as an input and produce the
        # concept predictions after applying the appropriate non-linearities
        self.x2c_model = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(latent_dim, n_concepts),
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
        if active_intervention_values is not None:
            self.active_intervention_values = torch.FloatTensor(
                active_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.active_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * (
                5.0 if not sigmoidal_prob else 1.0
            )
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.FloatTensor(
                inactive_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.inactive_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * (
                -5.0 if not sigmoidal_prob else 0.0
            )

        # For legacy purposes, we wrap the model around a torch.nn.Sequential
        # module
        self.sig = torch.nn.Sigmoid()
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

        # Construct the unsupervised capacity model which is a function of the
        # predicted concepts and the input features themselves
        if extra_dims > 0:
            if entanglement_mode == "concat":
                self.unsupervised_capacity_model = torch.nn.Linear(
                    n_concepts + latent_dim,
                    extra_dims,
                )
                self._residual_weights = self.unsupervised_capacity_model.weight
            elif entanglement_mode == "additive":
                # Then we will have one layer mapping the latent code to the
                # extra capacity and another layer mapping the concepts to
                # the extra capacity and then we will add them together
                self._latent_to_extra = torch.nn.Linear(
                    latent_dim,
                    extra_dims,
                )
                self._residual_weights = self._latent_to_extra.weight
                self.latent_to_extra = torch.nn.Sequential(
                    torch.nn.LeakyReLU(),
                    self._latent_to_extra,
                    self.bottleneck_nonlin,
                )
                self.concepts_to_extra = torch.nn.Sequential(
                    torch.nn.Linear(n_concepts, extra_dims),
                    self.bottleneck_nonlin,
                )
                self.unsupervised_capacity_model = lambda x: (
                    self.latent_to_extra(
                        x[:, n_concepts:]
                    ) +
                    self.concepts_to_extra(
                        x[:, :n_concepts]
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported entanglement mode '{entanglement_mode}'"
                )
        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                pos_weight=task_class_weights
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
        self.use_concept_groups = use_concept_groups
        self.training_intervention_prob = training_intervention_prob
        self.prior_loss_term = prior_loss_term
        self.prior_only_concepts = prior_only_concepts
        self._latent_code = None
        self.l2_reg_residual_weights = l2_reg_residual_weights
        self.entanglement_mode = entanglement_mode

    def _concept_intervention(
        self,
        c_pred,
        intervention_idxs=None,
        c_true=None,
    ):
        if (c_true is None) or (intervention_idxs is None):
            return c_pred
        intervention_idxs = self._standardize_indices(
            intervention_idxs=intervention_idxs,
            batch_size=c_pred.shape[0],
            device=c_pred.device,
        )
        intervention_idxs = intervention_idxs.to(c_pred.device).float()
        # Check whether the mask needs to be extended because of
        # extra dimensions
        concept_reprs = c_pred[:, :self.n_concepts]
        if not self.sigmoidal_prob:
            raise ValueError(
                "Interventions on EntangledHybridCBM only support sigmoidal "
                "concept probabilities."
            )
        new_bottleneck = intervention_idxs * c_true + (1 - intervention_idxs) * concept_reprs
        output = new_bottleneck
        if self.extra_dims:
            # Recompute the extra capacity based on the new concepts
            extra_capacity = self.unsupervised_capacity_model(
                torch.cat([new_bottleneck, self._latent_code], dim=-1)
            )
            output = torch.cat([new_bottleneck, extra_capacity], dim=-1)
        return output

    def _forward(
        self,
        x,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        output_latent=None,
        output_embeddings=False,
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            self._latent_code = self.latent_code_generator(x)
            latent = self._latent_code

        c_pred_logits = self.x2c_model(latent)
        c_sem = self.sig(c_pred_logits)

        if self.sigmoidal_prob:
            c_pred = c_sem
        elif self.bool:
            c_pred = (c_sem > 0.5).float()
        else:
            # Otherwise, the concept vector itself is not sigmoided
            # but the semantics
            c_pred = c_pred_logits

        if self.extra_dims > 0:
            extra_capacity = self.unsupervised_capacity_model(
                torch.cat([c_pred, latent], dim=-1)
            )
            c_pred = torch.cat([c_pred, extra_capacity], axis=-1)

        if output_embeddings or (
            (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        )):
            pos_embeddings = torch.ones(c_sem.shape).to(x.device)
            neg_embeddings = torch.zeros(c_sem.shape).to(x.device)
            if not (self.sigmoidal_prob or self.bool):
                if (
                    (self.active_intervention_values is not None) and
                    (self.inactive_intervention_values is not None)
                ):
                    active_intervention_values = \
                        self.active_intervention_values.to(
                            c_pred.device
                        )
                    pos_embeddings = torch.tile(
                        active_intervention_values,
                        (c.shape[0], 1),
                    ).to(active_intervention_values.device)
                    inactive_intervention_values = \
                        self.inactive_intervention_values.to(
                            c_pred.device
                        )
                    neg_embeddings = torch.tile(
                        inactive_intervention_values,
                        (c.shape[0], 1),
                    ).to(inactive_intervention_values.device)
                else:
                    out_embs = c_pred.detach().cpu().numpy()
                    for concept_idx in range(self.n_concepts):
                        pos_embeddings[:, concept_idx] = np.percentile(
                            out_embs[:, concept_idx],
                            95,
                        )
                        neg_embeddings[:, concept_idx] = np.percentile(
                            out_embs[:, concept_idx],
                            5,
                        )
            pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
            neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)

        # Now include any interventions that we may want to include
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            prior_distribution = self._prior_int_distribution(
                c=c,
                prob=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                competencies=competencies,
                prev_interventions=prev_interventions,
                train=train,
                horizon=1,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )
        else:
            c_int = c

        # Do training time interventions if requested:
        if train and (self.training_intervention_prob > 0.0) and (
            intervention_idxs is None
        ):
            intervention_idxs = torch.rand(
                (c_pred.shape[0], self.n_concepts)
            ).to(x.device) < self.training_intervention_prob

        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )
        # Make the downstream predictions
        y_pred = self.c2y_model(c_pred)

        tail_results = []
        if output_interventions:
            if intervention_idxs is None:
                intervention_idxs = None
            if isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)

        if output_latent:
            tail_results.append(latent)

        if output_embeddings:
            tail_results.append(pos_embeddings)
            tail_results.append(neg_embeddings)

        tail_results += self._extra_tail_results(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        return tuple([c_sem, c_pred, y_pred] + tail_results)

    def _extra_losses(
        self,
        x,
        y,
        c,
        y_pred,
        c_sem,
        c_pred,
        competencies=None,
        prev_interventions=None,
    ):
        loss = super()._extra_losses(
            x=x,
            y=y,
            c=c,
            y_pred=y_pred,
            c_sem=c_sem,
            c_pred=c_pred,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        if self.l2_reg_residual_weights > 0.0:
            if self.entanglement_mode == "concat":
                weights = self._residual_weights[:, self.n_concepts:]
            else:
               weights = self._residual_weights
            loss += self.l2_reg_residual_weights * torch.sum(
                weights ** 2
            )
        return loss



class LeakyReprCBM(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=0.01,
        task_loss_weight=1,
        sigmoidal_prob=True,

        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        weight_loss=None,
        task_class_weights=None,

        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        # New additions
        training_intervention_prob=0.0,
        prior_loss_term=0.0,

        top_k_accuracy=None,

        # Parameters for representation definition
        total_range_size=0.1,
        encode_concepts=False,
        encode_tasks=True,
        random_bucket_noise=True,
        classes_selected=1.0,
        concept_selected=1.0,
        max_repr_val=10000,
        prob_correct=1.0,
        prob_flips=0.0,
        leak_unprovided_concepts=False
    ):
        """
        TODO
        """
        # assert encode_concepts or encode_tasks, (
        #     "At least one of encode_concepts or encode_tasks must be True."
        # )

        self.total_range_size = total_range_size
        self.encode_concepts = encode_concepts
        self.leak_unprovided_concepts = leak_unprovided_concepts
        self.encode_tasks = encode_tasks
        self.random_bucket_noise = random_bucket_noise
        self.classes_selected = classes_selected
        self.concept_selected = concept_selected
        self.prob_correct = prob_correct
        self.prob_flips = prob_flips
        if sigmoidal_prob:
            self.max_repr_val = 1.0
            self.min_repr_val = 0.0

            active_intervention_values = None
            inactive_intervention_values = None
        else:
            self.max_repr_val = max_repr_val + 1
            self.min_repr_val = - self.max_repr_val
            active_intervention_values = [
                self.max_repr_val for _ in range(n_concepts)
            ]
            inactive_intervention_values = [
                self.min_repr_val for _ in range(n_concepts)
            ]

        super().__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,

            extra_dims=0,
            bool=False,
            sigmoidal_prob=sigmoidal_prob,
            sigmoidal_extra_capacity=False,
            bottleneck_nonlinear=None,
            output_latent=output_latent,
            x2c_model=x2c_model,
            c_extractor_arch=c_extractor_arch,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            use_concept_groups=use_concept_groups,
            training_intervention_prob=training_intervention_prob,
            prior_loss_term=prior_loss_term,
            prior_only_concepts=False,
            top_k_accuracy=top_k_accuracy,
        )




    # def _forward(
    #     self,
    #     x,
    #     intervention_idxs=None,
    #     competencies=None,
    #     prev_interventions=None,
    #     c=None,
    #     y=None,
    #     train=False,
    #     latent=None,
    #     output_latent=None,
    #     output_embeddings=False,
    #     output_interventions=None,
    # ):
    #     output_interventions = (
    #         output_interventions if output_interventions is not None
    #         else self.output_interventions
    #     )
    #     output_latent = (
    #         output_latent if output_latent is not None
    #         else self.output_latent
    #     )
    #     if isinstance(c, (tuple, list)):
    #         # Then we are working with a set of provided concepts and unprovided
    #         # concepts
    #         c, unprovided_concepts = c
    #     else:
    #         unprovided_concepts = None

    #     if c is None:
    #         raise ValueError(
    #             "FixedReprCBM requires concept labels to be provided "
    #             "during forward passes."
    #         )

    #     if self.leak_unprovided_concepts:
    #         assert unprovided_concepts is not None, (
    #             "leak_unprovided_concepts is True, but no unprovided "
    #             "concepts were given."
    #         )
    #         leak_c = unprovided_concepts
    #     else:
    #         leak_c = c


    #     # Now that we know we have both concept and task labels, we can
    #     # construct the fixed representations.
    #     # Let's start by calculating how many "buckets" we will use in the
    #     # ranges we were given
    #     self.n_buckets = 1
    #     if self.leak_unprovided_concepts:
    #         self.max_concept_idx = int(np.ceil(unprovided_concepts.shape[-1] * self.concept_selected))
    #     else:
    #         self.max_concept_idx = int(np.ceil(self.n_concepts * self.concept_selected))
    #     self.max_task_idx = int(np.ceil(self.n_tasks * self.classes_selected))
    #     if self.encode_concepts:
    #         self.n_buckets *= 2 ** self.max_concept_idx
    #     if self.encode_tasks:
    #         self.n_buckets *= self.max_task_idx
    #     # Now compute the size of each bucket
    #     self.pos_concepts_start_val = (
    #         (self.max_repr_val - self.total_range_size)
    #     )
    #     self.neg_concepts_start_val = self.min_repr_val

    #     self.bucket_size = self.total_range_size / self.n_buckets
    #     # Let's calculate how many different buckets we can have per concept so
    #     # far
    #     self.n_buckets_per_task = 2 ** self.max_concept_idx if self.encode_concepts else 1


    #     if y is None:
    #         raise ValueError(
    #             "FixedReprCBM requires task labels to be provided "
    #             "during forward passes."
    #         )

    #     # Now compute the bucket index for each example. This index will be
    #     # determined by the concept labels and the task labels. Each combination
    #     # of concept and task labels will correspond to a unique bucket.
    #     # Then we will generate a represention for each concept such that, if
    #     # the concept is on, we assign it a value in [1-self.total_range_size, 1]
    #     # in the bucket assigned to the example, and if the concept is off,
    #     # we assign it a value in [0, self.total_range_size] in the bucket
    #     # assigned to the example.
    #     batch_size = c.shape[0]
    #     bucket_indices = torch.zeros((batch_size, ), dtype=torch.long).to(x.device)
    #     if self.encode_concepts:
    #         # We add the binary representation of the concept vector to the
    #         # bucket index
    #         for concept_idx in range(self.max_concept_idx):
    #             bucket_indices += (
    #                 leak_c[:, concept_idx].long() *
    #                 (2 ** concept_idx)
    #             )

    #     if self.encode_tasks:
    #         # and next we shift the current bucket index based on the ground
    #         # truth task labels (where y is a [B] vector of class indices)
    #         # however, we only do this for the first self.classes_selected
    #         # classes
    #         if self.prob_flips > 0:
    #             # Then we randomly change some of the task labels by setting
    #             # a task label in y to a random label in {0, ..., n_tasks - 1}
    #             # with probability self.prob_flips
    #             random_labels = torch.randint(
    #                 low=0,
    #                 high=self.n_tasks,
    #                 size=y.shape,
    #             ).to(y.device)
    #             y_flip_mask = (
    #                 torch.rand(y.shape).to(y.device) <= self.prob_flips
    #             ).long()
    #             ##print("y_flip_mask[:5] =", y_flip_mask[:5])
    #             leak_y = y * (1 - y_flip_mask) + random_labels * y_flip_mask
    #         else:
    #             leak_y = y
    #         ##print("leak_y[:5] =", leak_y[:5])
    #         bucket_indices += (leak_y.long().clamp(
    #             max=self.max_task_idx
    #         ) * self.n_buckets_per_task)
    #         # All classes above the selected ones are mapped to the last
    #         # bucket for the selected classes

    #     # Now we can compute the lower bound of each bucket
    #     bucket_lower_bounds = bucket_indices.float() * self.bucket_size

    #     # We can compute the representations for each concept
    #     reprs = torch.zeros((batch_size, self.n_concepts)).to(x.device)
    #     # We will flip concepts with probability 1 - self.prob_correct
    #     if self.prob_correct < 1.0:
    #         flip_mask = (
    #             torch.rand((batch_size, self.n_concepts)).to(x.device) >
    #             self.prob_correct
    #         ).float()
    #         used_concepts = c * (1 - flip_mask) + (1 - c) * flip_mask
    #     else:
    #         used_concepts = c

    #     for concept_idx in range(self.n_concepts):
    #         concept_on_values = self.pos_concepts_start_val + bucket_lower_bounds
    #         concept_off_values = self.neg_concepts_start_val + bucket_lower_bounds
    #         reprs[:, concept_idx] = (
    #             used_concepts[:, concept_idx] * concept_on_values +
    #             (1 - used_concepts[:, concept_idx]) * concept_off_values
    #         )
    #     if self.random_bucket_noise:
    #         reprs += (
    #             torch.rand((batch_size, self.n_concepts)).to(x.device) * self.bucket_size
    #         )
    #     else:
    #         reprs += (self.bucket_size / 2.0)
    #     ##print("reprs[:5] =", reprs[:5])

    #     # Now pass the representations through the concept predictor
    #     # and the label predictor
    #     if self.sigmoidal_prob:
    #         c_sem = reprs
    #         c_pred = reprs
    #     else:
    #         c_pred = reprs
    #         c_sem = self.sig(reprs)

    #     if output_embeddings or (
    #         (intervention_idxs is None) and (c is not None) and (
    #         self.intervention_policy is not None
    #     )):
    #         pos_embeddings = torch.ones(c_sem.shape).to(x.device)
    #         neg_embeddings = torch.zeros(c_sem.shape).to(x.device)
    #         pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
    #         neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)

    #     # Now include any interventions that we may want to include
    #     if (intervention_idxs is None) and (c is not None) and (
    #         self.intervention_policy is not None
    #     ):
    #         prior_distribution = self._prior_int_distribution(
    #             c=c,
    #             prob=c_sem,
    #             pos_embeddings=pos_embeddings,
    #             neg_embeddings=neg_embeddings,
    #             competencies=competencies,
    #             prev_interventions=prev_interventions,
    #             train=train,
    #             horizon=1,
    #         )
    #         intervention_idxs, c_int = self.intervention_policy(
    #             x=x,
    #             c=c,
    #             pred_c=c_sem,
    #             y=y,
    #             competencies=competencies,
    #             prev_interventions=prev_interventions,
    #             prior_distribution=prior_distribution,
    #         )
    #     else:
    #         c_int = c

    #     if train and (self.training_intervention_prob > 0.0) and (
    #         intervention_idxs is None
    #     ):
    #         intervention_idxs = torch.rand(
    #             (c_pred.shape[0], self.n_concepts)
    #         ).to(x.device) < self.training_intervention_prob

    #     c_pred = self._concept_intervention(
    #         c_pred=c_pred,
    #         intervention_idxs=intervention_idxs,
    #         c_true=c_int,
    #     )

    #     y_pred = self.c2y_model(c_pred)

    #     tail_results = []
    #     if output_interventions:
    #         if intervention_idxs is None:
    #             intervention_idxs = None
    #         if isinstance(intervention_idxs, np.ndarray):
    #             intervention_idxs = torch.FloatTensor(
    #                 intervention_idxs
    #             ).to(x.device)
    #         tail_results.append(intervention_idxs)
    #     if output_latent:
    #         tail_results.append(latent)
    #     if output_embeddings:
    #         tail_results.append(pos_embeddings)
    #         tail_results.append(neg_embeddings)
    #     tail_results += self._extra_tail_results(
    #         x=x,
    #         y=y,
    #         c=c,
    #         c_sem=c_sem,
    #         competencies=competencies,
    #         prev_interventions=prev_interventions,
    #     )
    #     return tuple([c_sem, c_pred, y_pred] + tail_results)




    def _forward(
        self,
        x,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        output_latent=None,
        output_embeddings=False,
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if isinstance(c, (tuple, list)) and len(c) == 2:
            # Then we are working with a set of provided concepts and unprovided
            # concepts
            c, unprovided_concepts = c
        else:
            unprovided_concepts = None

        if c is None:
            raise ValueError(
                "FixedReprCBM requires concept labels to be provided "
                "during forward passes."
            )

        if self.leak_unprovided_concepts:
            assert unprovided_concepts is not None, (
                "leak_unprovided_concepts is True, but no unprovided "
                "concepts were given."
            )
            leak_c = unprovided_concepts
        else:
            leak_c = c


        if y is None:
            raise ValueError(
                "FixedReprCBM requires task labels to be provided "
                "during forward passes."
            )
        if self.prob_flips > 0:
            # Then we randomly change some of the task labels by setting
            # a task label in y to a random label in {0, ..., n_tasks - 1}
            # with probability self.prob_flips
            random_labels = torch.randint(
                low=0,
                high=self.n_tasks,
                size=y.shape,
            ).to(y.device)
            y_flip_mask = (
                torch.rand(y.shape).to(y.device) <= self.prob_flips
            ).long()
            ##print("y_flip_mask[:5] =", y_flip_mask[:5])
            leak_y = y * (1 - y_flip_mask) + random_labels * y_flip_mask
        else:
            leak_y = y

        if self.prob_correct < 1.0:
            flip_mask = (
                torch.rand((x.shape[0], self.n_concepts)).to(x.device) >
                self.prob_correct
            ).float()
            used_concepts = c * (1 - flip_mask) + (1 - c) * flip_mask
        else:
            used_concepts = c

        # reprs will modify the concepts representations so that the binary
        # encoding of the ground-truth label is encoded as part of a fluctuation
        # on the concept activation pattern
        c_sem = used_concepts
        c_pred = used_concepts.clone()
        # encode the ground-truth task labels using a binary representation
        # with n_concept bits at most
        batch_size = x.shape[0]
        if self.sigmoidal_prob:
            for i in range(batch_size):
                task_label = leak_y[i].long().item()
                for concept_idx in range(self.n_concepts):
                    bit_value = (task_label >> concept_idx) & 1
                    if bit_value == 1:
                        shift = self.max_repr_val * torch.rand(1).item() if self.random_bucket_noise else self.max_repr_val
                        if c_pred[i, concept_idx] == 1:
                            c_pred[i, concept_idx] -= shift
                        else:
                            c_pred[i, concept_idx] += shift
        else:
            for i in range(batch_size):
                task_label = leak_y[i].long().item()
                for concept_idx in range(self.n_concepts):
                    c_pred[i, concept_idx] = self.min_repr_val if c_sem[i, concept_idx] == 0 else self.max_repr_val
                    # if self.random_bucket_noise:
                    #     c_pred[i, concept_idx] += torch.randn(1).item()
                    bit_value = (task_label >> concept_idx) & 1
                    if bit_value == 1:
                        # Then shift the representation by 0.5
                        if isinstance(self.encode_tasks, (float, int)):
                            pos_shift = neg_shift = self.encode_tasks
                        elif isinstance(self.encode_tasks, (list, tuple)):
                            pos_shift = self.encode_tasks[0]
                            neg_shift = self.encode_tasks[1]
                        else:
                            shift = self.max_repr_val/2 * torch.rand(1).item() if self.random_bucket_noise else self.max_repr_val/2
                        # c_pred[i, concept_idx] += shift
                        if c_sem[i, concept_idx] == 1:
                            c_pred[i, concept_idx] += pos_shift
                        else:
                            c_pred[i, concept_idx] -= neg_shift


        if output_embeddings or (
            (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        )):
            pos_embeddings = torch.ones(c_sem.shape).to(x.device)
            neg_embeddings = torch.zeros(c_sem.shape).to(x.device)
            pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
            neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)

        # Now include any interventions that we may want to include
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            prior_distribution = self._prior_int_distribution(
                c=c,
                prob=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                competencies=competencies,
                prev_interventions=prev_interventions,
                train=train,
                horizon=1,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )
        else:
            c_int = c

        if train and (self.training_intervention_prob > 0.0) and (
            intervention_idxs is None
        ):
            intervention_idxs = torch.rand(
                (c_pred.shape[0], self.n_concepts)
            ).to(x.device) < self.training_intervention_prob

        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )

        y_pred = self.c2y_model(c_pred)

        tail_results = []
        if output_interventions:
            if intervention_idxs is None:
                intervention_idxs = None
            if isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(pos_embeddings)
            tail_results.append(neg_embeddings)
        tail_results += self._extra_tail_results(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        return tuple([c_sem, c_pred, y_pred] + tail_results)