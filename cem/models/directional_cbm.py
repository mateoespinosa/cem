import sklearn.metrics
import torch
import pytorch_lightning as pl
from torchvision.models import resnet50
import numpy as np

import cem.train.utils as utils
from cem.models.cbm import ConceptBottleneckModel
from cem.models.construction import LambdaLayer


################################################################################
## Helper Classes
################################################################################


class MixedActivation(torch.nn.Module):
    """
    A custom PyTorch layer that applies different activation functions to
    different segments of the input tensor.

    Args:
        activations (list of torch.nn.Module or callable):
            List of activation functions (e.g., [torch.nn.ReLU(),
            torch.nn.Sigmoid(), torch.nn.Tanh()]).
        ranges (list of int):
            List of sizes for each activation segment. The sum of ranges must
            match the input dimension.
    """
    def __init__(self, activations, ranges):
        super(MixedActivation, self).__init__()

        assert len(activations) == len(ranges), \
            "The number of activations must match the number of ranges."

        self.activations = torch.nn.ModuleList(
            [
                act if isinstance(act, torch.nn.Module) else LambdaLayer(act)
                for act in activations
            ]
        )
        self.ranges = ranges

    def forward(self, x):
        """
        Applies each activation to its corresponding input slice and
        concatenates the results.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        assert x.size(1) == sum(self.ranges), (
            f"Input feature size ({x.size(1)}) does not match sum "
            f"of ranges ({sum(self.ranges)})."
        )

        outputs = []
        start = 0
        for act, r in zip(self.activations, self.ranges):
            end = start + r
            segment = x[:, start:end]
            outputs.append(act(segment))
            start = end

        return torch.cat(outputs, dim=1)

################################################################################
## Directional CBM
################################################################################


class DirectionalCBM(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=1,
        task_loss_weight=1,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

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

        top_k_accuracy=None,

        # Model-specific arguments
        training_edge_drop_prob=None,
        c2y_pred_weight=0,
        conditional_model=None,
        num_inference_passes=1,
    ):
        """
        TODO
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.latent_code_model = c_extractor_arch(output_dim=None)
        self._intervention_idxs = None

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.top_k_accuracy = top_k_accuracy

        # First build the model that maps the latent code to the
        # initial set of concept + task predictions
        self.bottleneck_size = self.n_concepts + self.n_tasks
        self.act_layer = MixedActivation(
            [torch.nn.Sigmoid(), torch.nn.Softmax()],
            [self.n_concepts, self.n_tasks],
        )
        self.initial_prob_model = torch.nn.Sequential(
            torch.nn.Linear(
                list(
                    self.latent_code_model.modules()
                )[-1].out_features,
                # Two as each concept will have a positive and a
                # negative embedding portion which are later mixed
                self.bottleneck_size,
            ),
            self.act_layer,
        )

        # Next build the conditional masked model that, given a set of
        # potentially masked samples
        if conditional_model is None:
            self.cond_model = torch.nn.Sequential(
                torch.nn.Linear(
                    # The input will be the previous distribution + the mask
                    # indicating which samples have been set to their average
                    # values
                    self.bottleneck_size + self.bottleneck_size,
                    self.bottleneck_size,
                ),
                self.act_layer,
            )
        else:
            self.cond_model = conditional_model

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.n_tasks = n_tasks
        self.use_concept_groups = use_concept_groups

        # Model-specific bits
        self.training_edge_drop_prob = training_edge_drop_prob
        self.c2y_pred_weight = c2y_pred_weight
        self.num_inference_passes = num_inference_passes



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
        # First generate the latent code
        if latent is None:
            latent = self.latent_code_model(x)

        # Next generate the initial probability distributions:
        initial_probs = self.initial_prob_model(latent)

        # Produce the initial concept predictions
        c_sem = initial_probs[:, :self.n_concepts]
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
        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )

        # Now that we have performed any interventions, time to perform some
        # updates on the probability distribution
        y_pred = self.c2y_model((c_pred > 0.5).float())

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