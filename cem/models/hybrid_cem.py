import numpy as np
import pytorch_lightning as pl
import scipy
import sklearn.metrics
import torch

from torchvision.models import resnet50

from cem.metrics.accs import compute_accuracy
from cem.models.cem import ConceptEmbeddingModel
import cem.train.utils as utils



################################################################################
## OUR MODEL
################################################################################


class HybridConceptEmbeddingModel(ConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        constant_emb_proportion=0.5,
        training_intervention_prob=0.25,
        dyn_embedding_activation="leakyrelu",
        const_embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,
        contrastive_anchors=False,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        tau=1,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,
    ):
        """
        Constructs a Hybrid Concept Embedding Model (H-CEM).

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the H-CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param float training_intervention_prob: RandInt probability. Defaults
            to 0.25.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            H-CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the H-CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this H-CEM's inputs to the latent space of the
            requested size.

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
            parameter is important when intervening in CEMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts.
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
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        self.contrastive_anchors = contrastive_anchors

        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

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
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy

        self.emb_size = emb_size

        if self.contrastive_anchors:
            self.concept_embeddings = torch.nn.Parameter(
                torch.rand(self.n_concepts, 2, self.emb_size),
            )
            self.contrastive_scale = torch.nn.Parameter(
                torch.rand((self.n_concepts,))
            )
        else:
            self.constant_emb_dims = int(np.ceil(
                constant_emb_proportion * self.emb_size
            ))
            if self.constant_emb_dims != 0:
                self.concept_embeddings = torch.nn.Parameter(
                    torch.rand(self.n_concepts, 2, self.constant_emb_dims),
                )

        for i in range(n_concepts):
            if const_embedding_activation is None:
                self.const_emb_act = lambda x: x
            elif const_embedding_activation == "sigmoid":
                self.const_emb_act = torch.nn.Sigmoid()
            elif const_embedding_activation == "leakyrelu":
                self.const_emb_act = torch.nn.LeakyReLU()
            elif const_embedding_activation == "relu":
                self.const_emb_act = torch.nn.ReLU()
            else:
                raise ValueError(
                    f'Unsupported embedding activation "{const_embedding_activation}"'
                )
            if dyn_embedding_activation is None:
                self.dyn_emb_act = lambda x: x
            elif dyn_embedding_activation == "sigmoid":
                self.dyn_emb_act = torch.nn.Sigmoid()
            elif dyn_embedding_activation == "leakyrelu":
                self.dyn_emb_act = torch.nn.LeakyReLU()
            elif dyn_embedding_activation == "relu":
                self.dyn_emb_act = torch.nn.ReLU()
            else:
                raise ValueError(
                    f'Unsupported embedding activation "{const_embedding_activation}"'
                )
            if self.contrastive_anchors:
                self.concept_context_generators.append(
                        torch.nn.Sequential(*[
                            torch.nn.Linear(
                                list(
                                    self.pre_concept_model.modules()
                                )[-1].out_features,
                                # Two as each concept will have a positive and a
                                # negative embedding portion which are later mixed
                                2 * emb_size,
                            ),
                        ])
                    )
            else:
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * (emb_size - self.constant_emb_dims),
                        ),
                    ])
                )
                if self.shared_prob_gen and (
                    len(self.concept_prob_generators) == 0
                ):
                    # Then we will use one and only one probability generator
                    # which will be shared among all concepts. This will force
                    # concept embedding vectors to be pushed into the same
                    # latent space
                    self.concept_prob_generators.append(torch.nn.Linear(
                        2 * emb_size,
                        1,
                    ))
                elif not self.shared_prob_gen:
                    self.concept_prob_generators.append(torch.nn.Linear(
                        2 * emb_size,
                        1,
                    ))

        if c2y_model is None:
            # Else we construct it here directly
            units = [
                n_concepts * emb_size
            ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()

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
        self.n_tasks = n_tasks
        self.tau = tau
        self.use_concept_groups = use_concept_groups

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        dynamic_scalings=None,
        **kwargs,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            # Then time to mix!
            bottleneck = (
                pos_embeddings * torch.unsqueeze(prob, dim=-1) +
                neg_embeddings * (1 - torch.unsqueeze(prob, dim=-1))
            )
            return prob, intervention_idxs, bottleneck
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        output = prob * (1 - intervention_idxs) + intervention_idxs * c_true
        if self.contrastive_anchors:
            pos_anchors = torch.stack(
                [
                    self.concept_embeddings[:, 0, :]
                    for _ in range(intervention_idxs.shape[0])
                ],
                axis=0,
            )
            neg_anchors = torch.stack(
                [
                    self.concept_embeddings[:, 1, :]
                    for _ in range(intervention_idxs.shape[0])
                ],
                axis=0,
            )
            extended_intervention_idxs = torch.unsqueeze(
                intervention_idxs,
                dim=-1,
            )
            pos_embeddings = (
                (1 - extended_intervention_idxs) * pos_embeddings +
                extended_intervention_idxs * (
                    pos_anchors * dynamic_scalings
                )
            )
            neg_embeddings = (
                (1 - extended_intervention_idxs) * neg_embeddings +
                extended_intervention_idxs * (
                    neg_anchors * dynamic_scalings
                )
            )
        # Then time to mix!
        bottleneck = (
            pos_embeddings * torch.unsqueeze(output, dim=-1) +
            neg_embeddings * (1 - torch.unsqueeze(output, dim=-1))
        )
        return output, intervention_idxs, bottleneck

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []
            scalings = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                context = context_gen(pre_c)
                if self.contrastive_anchors:
                    const_embs = self.dyn_emb_act(
                        context[:, :self.emb_size]
                    )
                    dyn_embs = self.const_emb_act(
                        context[:, self.emb_size:]
                    )
                    anchors = torch.stack(
                        [
                            self.concept_embeddings[i, :, :]
                            for _ in range(context.shape[0])
                        ],
                        axis=0,
                    )
                    pos_anchors = anchors[:, 0, :]
                    neg_anchors = anchors[:, 1, :]
                    prob = self.sig(
                        self.contrastive_scale[i] * (
                            (neg_anchors - const_embs).pow(2).sum(-1).sqrt() -
                            (pos_anchors - const_embs).pow(2).sum(-1).sqrt()
                        )
                    )
                    prob = torch.unsqueeze(prob, dim=-1)
                    context = torch.cat(
                        [dyn_embs * pos_anchors, dyn_embs * neg_anchors],
                        dim=-1,
                    )
                    scalings.append(torch.unsqueeze(dyn_embs, dim=1))
                else:
                    if self.shared_prob_gen:
                        prob_gen = self.concept_prob_generators[0]
                    else:
                        prob_gen = self.concept_prob_generators[i]
                    dynamic_emb_dims = self.emb_size - self.constant_emb_dims
                    context = torch.cat(
                        [
                            self.dyn_emb_act(torch.unsqueeze(
                                context[:, :dynamic_emb_dims],
                                axis=1,
                            )),
                            self.dyn_emb_act(torch.unsqueeze(
                                context[:, dynamic_emb_dims:],
                                axis=1,
                            )),
                        ],
                        dim=1,
                    )
                    constant_compontents = torch.stack(
                        [
                            self.concept_embeddings[i, :, :]
                            for _ in range(context.shape[0])
                        ],
                        axis=0,
                    )
                    context = torch.cat(
                        [context, constant_compontents],
                        dim=-1,
                    )
                    context = torch.cat(
                        [context[:, 0, :], context[:, 1, :]],
                        dim=-1
                    )
                    prob = self.sig(prob_gen(context))
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(prob)
            c_sem = torch.cat(c_sem, dim=-1)
            contexts = torch.cat(contexts, dim=1)
            latent = contexts, c_sem
            if self.contrastive_anchors:
                scalings = torch.cat(scalings, dim=1)
        else:
            if self.contrastive_anchors:
                contexts, c_sem, scalings = latent
            else:
                contexts, c_sem = latent
        pos_embeddings = contexts[:, :, :self.emb_size]
        neg_embeddings = contexts[:, :, self.emb_size:]
        if self.contrastive_anchors:
            return c_sem, pos_embeddings, neg_embeddings, {
                "dynamic_scalings": scalings,
            }
        return c_sem, pos_embeddings, neg_embeddings, {}





class MultiConceptEmbeddingModel(ConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        n_discovered_embs=4,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        concept_loss_weight=1,
        task_loss_weight=1,
        contrastive_loss_weight=0,
        mix_ground_truth_embs=True,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        tau=1,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,
    ):
        """
        Constructs a Hybrid Concept Embedding Model (H-CEM).

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the H-CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param float training_intervention_prob: RandInt probability. Defaults
            to 0.25.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            H-CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the H-CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this H-CEM's inputs to the latent space of the
            requested size.

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
            parameter is important when intervening in CEMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts.
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
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        self.mix_ground_truth_embs = mix_ground_truth_embs

        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

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
        self.contrastive_loss_weight = contrastive_loss_weight
        if self.contrastive_loss_weight != 0:
            self.contrastive_target = -torch.ones([])
        self.contrastive_loss_fn = torch.nn.CosineEmbeddingLoss(
            margin=0.0,
            size_average=None,
            reduce=None,
            reduction='mean',
        )
        self.top_k_accuracy = top_k_accuracy

        self.emb_size = emb_size
        self.concept_embeddings = torch.nn.Parameter(
            torch.rand(self.n_concepts, 2, self.emb_size),
        )
        self.n_discovered_embs = n_discovered_embs
        if self.n_discovered_embs != 0:
            self.discovered_concept_embeddings = torch.nn.Parameter(torch.rand(
                self.n_concepts,
                self.n_discovered_embs,
                2,
                self.emb_size,
            ))
        self.contrastive_scale = torch.nn.Parameter(
            torch.rand((self.n_concepts,))
        )

        self.concept_emb_generators = torch.nn.ModuleList()
        self.concept_mixture_models = torch.nn.ModuleList()
        for i in range(n_concepts):
            if embedding_activation is None:
                emb_act = torch.nn.Identity()
            elif embedding_activation == "sigmoid":
                emb_act = torch.nn.Sigmoid()
            elif embedding_activation == "leakyrelu":
                emb_act = torch.nn.LeakyReLU()
            elif embedding_activation == "relu":
                emb_act = torch.nn.ReLU()
            else:
                raise ValueError(
                    f'Unsupported embedding activation "{embedding_activation}"'
                )
            self.concept_emb_generators.append(
                torch.nn.Sequential(*[
                    torch.nn.Linear(
                        list(
                            self.pre_concept_model.modules()
                        )[-1].out_features,
                        emb_size,
                    ),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(
                        emb_size,
                        emb_size,
                    ),
                    emb_act,
                ])
            )
            if self.n_discovered_embs != 0:
                self.concept_mixture_models.append(torch.nn.Sequential(*[
                    torch.nn.Linear(
                        list(
                            self.pre_concept_model.modules()
                        )[-1].out_features,
                        self.n_discovered_embs,
                    ),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(
                        self.n_discovered_embs,
                        self.n_discovered_embs,
                    ),
                    torch.nn.Sigmoid(),
                ]))

        if c2y_model is None:
            # Else we construct it here directly
            units = [
                self.n_concepts * self.emb_size * (
                    2 if self.n_discovered_embs > 0 else 1
                )
            ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()

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
        self.n_tasks = n_tasks
        self.tau = tau
        self.use_concept_groups = use_concept_groups

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
        current_loss = 0.0
        if self.contrastive_loss_weight == 0 or (
            self.n_discovered_embs == 0
        ):
            return current_loss

        extra_losses = []
        for concept_idx in range(self.n_concepts):
            for extra_concept_idx in range(self.n_discovered_embs):
                for selected_idx_1 in [0, 1]:
                    for selected_idx_2 in [0, 1]:
                        extra_losses.append(self.contrastive_loss_fn(
                            self.concept_embeddings[concept_idx, selected_idx_1, :],
                            self.discovered_concept_embeddings[
                                concept_idx,
                                extra_concept_idx,
                                selected_idx_2,
                                :
                            ],
                            self.contrastive_target.to(
                                self.discovered_concept_embeddings.device
                            ),
                        ))
                for extra_concept_idx_2 in range(extra_concept_idx + 1, self.n_discovered_embs):
                    for selected_idx_1 in [0, 1]:
                        for selected_idx_2 in [0, 1]:
                            extra_losses.append(self.contrastive_loss_fn(
                                self.discovered_concept_embeddings[
                                    concept_idx,
                                    extra_concept_idx,
                                    selected_idx_1,
                                    :
                                ],
                                self.discovered_concept_embeddings[
                                    concept_idx,
                                    extra_concept_idx_2,
                                    selected_idx_2,
                                    :
                                ],
                                self.contrastive_target.to(
                                    self.discovered_concept_embeddings.device
                                ),
                            ))
        total_loss = 0.0
        for loss in extra_losses:
            total_loss += loss
        return current_loss + self.contrastive_loss_weight * total_loss/len(extra_losses)

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        out_embeddings=None,
        pred_concepts=None,
        dynamic_mixtures=None,
        **kwargs,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            if dynamic_mixtures is None:
                bottleneck = torch.flatten(
                    pred_concepts,
                    start_dim=1,
                )
            else:
                bottleneck = torch.flatten(
                    torch.cat(
                        [pred_concepts, dynamic_mixtures],
                        dim=-1,
                    ),
                    start_dim=1,
                )
            return prob, intervention_idxs, bottleneck
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        output = prob * (1 - intervention_idxs) + intervention_idxs * c_true

        # [Shape: (1, n_concepts, emb_size)]
        pos_anchors = self.concept_embeddings[:, 0, :].unsqueeze(0)
        # [Shape: (1, n_concepts, emb_size)]
        neg_anchors = self.concept_embeddings[:, 1, :].unsqueeze(0)
        # [Shape: (B, n_concepts, 1)]
        extended_intervention_idxs = torch.unsqueeze(
            intervention_idxs,
            dim=-1,
        )
        # [Shape: (B, n_concepts, 1)]
        extended_c_true = c_true.unsqueeze(-1)
        # [Shape: (B, n_concepts, emb_size)]
        ground_truth_anchors = pos_anchors * extended_c_true + (1 - extended_c_true) * neg_anchors
        # [Shape: (B, n_concepts, emb_size)]
        pred_concepts = (
            (1 - extended_intervention_idxs) * pred_concepts +
            extended_intervention_idxs * ground_truth_anchors
        )
        # Then time to mix!
        if dynamic_mixtures is None:
            bottleneck = torch.flatten(
                pred_concepts,
                start_dim=1,
            )
        else:
            # [Shape: (B, n_concepts, 2*emb_size)]
            bottleneck = torch.flatten(
                torch.cat(
                    [pred_concepts, dynamic_mixtures],
                    dim=-1,
                ),
                start_dim=1,
            )
        return output, intervention_idxs, bottleneck

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            c_sem = []
            dynamic_mixtures = []
            pred_concepts = []

            # First predict all the concept probabilities
            for i, concept_emb_generator in enumerate(
                self.concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                pred_concept_embeddings = concept_emb_generator(pre_c)
                # [Shape: (1, emb_size)]
                anchor_concept_pos_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 0, :],
                    dim=0,
                )
                # [Shape: (1, emb_size)]
                anchor_concept_neg_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 1, :],
                    dim=0,
                )
                # [Shape: (B)]
                prob = self.sig(
                    self.contrastive_scale[i] * (
                        (anchor_concept_neg_emb - pred_concept_embeddings).pow(2).sum(-1).sqrt() -
                        (anchor_concept_pos_emb - pred_concept_embeddings).pow(2).sum(-1).sqrt()
                    )
                )
                # [Shape: (B, 1)]
                prob = torch.unsqueeze(prob, dim=-1)
                if self.mix_ground_truth_embs:
                    pred_concept_embeddings = prob * anchor_concept_pos_emb + (1 - prob) * anchor_concept_neg_emb

                if self.n_discovered_embs != 0:
                    # Now compute the dynamically mixed set of discovered embeddings
                    # [Shape: (B, n_discovered_embs)]
                    mixture_scales = self.concept_mixture_models[i](pre_c)
                    # [Shape: (B, n_discovered_embs, 1)]
                    mixture_scales = torch.unsqueeze(mixture_scales, dim=-1)
                    # [Shape: (n_discovered_embs, emb_size)]
                    dyn_pos_embeddings = self.discovered_concept_embeddings[
                        i,
                        :,
                        0,
                        :,
                    ]
                    # [Shape: (1, n_discovered_embs, emb_size)]
                    dyn_pos_embeddings = dyn_pos_embeddings.unsqueeze(
                        0
                    )
                    # [Shape: (n_discovered_embs, emb_size)]
                    dyn_neg_embeddings = self.discovered_concept_embeddings[
                        i,
                        :,
                        1,
                        :,
                    ]
                    # [Shape: (1, n_discovered_embs, emb_size)]
                    dyn_neg_embeddings = dyn_neg_embeddings.unsqueeze(
                        0
                    )

                    # And we do the mixing by accumulating the sums over all the
                    # discovered concepts
                    # [Shape: (B, emb_size)]
                    mixed_discovered_concept = (
                        mixture_scales[:, 0, :] * dyn_pos_embeddings[:, 0, :] +
                        (1 - mixture_scales[:, 0, :]) * dyn_neg_embeddings[:, 0, :]
                    )
                    for con_idx in range(1, self.n_discovered_embs):
                        mixed_discovered_concept += (
                            mixture_scales[:, con_idx, :] * dyn_pos_embeddings[:, con_idx, :] +
                            (1 - mixture_scales[:, con_idx, :]) * dyn_neg_embeddings[:, con_idx, :]
                        )
                    dynamic_mixtures.append(
                        torch.unsqueeze(mixed_discovered_concept, dim=1)
                    )
                c_sem.append(prob)
                pred_concepts.append(
                    torch.unsqueeze(pred_concept_embeddings, dim=1)
                )
            c_sem = torch.cat(c_sem, dim=-1)
            pred_concepts = torch.cat(pred_concepts, dim=1)
            if self.n_discovered_embs != 0:
                dynamic_mixtures = torch.cat(dynamic_mixtures, dim=1)
            else:
                dynamic_mixtures = None
            latent = c_sem, pred_concepts, dynamic_mixtures
        else:
            c_sem, pred_concepts, dynamic_mixtures = latent
        return c_sem, None, None, {
            "pred_concepts": pred_concepts,
            "dynamic_mixtures": dynamic_mixtures,
        }

def log(x):
    return torch.log(x + 1e-6)

def Gaussian_CDF(x): #change from erf function
    return 0.5 * (1. + torch.erf(x / np.sqrt(2.)))


def _binary_entropy(probs):
    return -probs * torch.log2(probs) - (1 - probs) * torch.log2(1 - probs)

class MixingConceptEmbeddingModel(ConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        n_discovered_concepts=4,
        training_intervention_prob=0.25,
        dyn_training_intervention_prob=0,
        embedding_activation="leakyrelu",
        concept_loss_weight=1,
        task_loss_weight=1,
        contrastive_loss_weight=0,
        intermediate_task_concept_loss=0,
        intervention_task_discount=1,
        discovered_probs_entropy=0,

        mix_ground_truth_embs=True,
        shared_emb_generator=False,
        normalize_embs=False,
        sample_probs=False,
        cov_mat=None,
        cond_discovery=False,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        tau=1,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.dyn_training_intervention_prob = dyn_training_intervention_prob
        self.output_latent = output_latent
        self.mix_ground_truth_embs = mix_ground_truth_embs
        self.normalize_embs = normalize_embs
        self.sample_probs = sample_probs
        self.intervention_task_discount = intervention_task_discount
        self.intermediate_task_concept_loss = intermediate_task_concept_loss
        self.discovered_probs_entropy = discovered_probs_entropy
        if cov_mat is None:
            cov_mat = np.eye(n_concepts, dtype=np.float32)
        self.cov_mat = cov_mat
        self.L = torch.tensor(
            scipy.linalg.cholesky(self.cov_mat, lower=True).astype(
                np.float32
            )
        )

        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)
        if self.dyn_training_intervention_prob != 0:
            self.dyn_ones = torch.ones(n_discovered_concepts)

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
        self.contrastive_loss_weight = contrastive_loss_weight
        if self.contrastive_loss_weight != 0:
            # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.contrastive_target = torch.zeros([])
            self.contrastive_target = -torch.ones([])
        self.contrastive_loss_fn = torch.nn.CosineEmbeddingLoss(
            margin=0.0,
            size_average=None,
            reduce=None,
            reduction='mean',
        )
        # self.contrastive_loss_fn = lambda x1, x2, *args, **kwargs: (x1 - x2).pow(2).sum(-1)
        self.top_k_accuracy = top_k_accuracy

        self.emb_size = emb_size
        self.concept_embeddings = torch.nn.Parameter(
            torch.rand(self.n_concepts, 2, self.emb_size),
        )
        self.n_discovered_concepts = n_discovered_concepts
        if self.n_discovered_concepts != 0:
            self.discovered_concept_embeddings = torch.nn.Parameter(torch.rand(
                self.n_discovered_concepts,
                2,
                self.emb_size,
            ))
            self.discovered_contrastive_scale = torch.nn.Parameter(
                torch.rand((self.n_discovered_concepts,))
            )
        self.contrastive_scale = torch.nn.Parameter(
            torch.rand((self.n_concepts,))
        )

        self.concept_emb_generators = torch.nn.ModuleList()
        self.discovered_concept_emb_generators = torch.nn.ModuleList()
        self.shared_emb_generator = shared_emb_generator
        self.cond_discovery = cond_discovery
        for i in range(n_concepts):
            if embedding_activation is None:
                emb_act = torch.nn.Identity()
            elif embedding_activation == "sigmoid":
                emb_act = torch.nn.Sigmoid()
            elif embedding_activation == "leakyrelu":
                emb_act = torch.nn.LeakyReLU()
            elif embedding_activation == "relu":
                emb_act = torch.nn.ReLU()
            else:
                raise ValueError(
                    f'Unsupported embedding activation "{embedding_activation}"'
                )
            if self.shared_emb_generator:
                if len(self.concept_emb_generators) == 0:
                    self.concept_emb_generators.append(
                        torch.nn.Sequential(*[
                            torch.nn.Linear(
                                list(
                                    self.pre_concept_model.modules()
                                )[-1].out_features,
                                emb_size,
                            ),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(
                                emb_size,
                                emb_size,
                            ),
                            emb_act,
                        ])
                    )
                else:
                    self.concept_emb_generators.append(
                        self.concept_emb_generators[0]
                    )
            else:
                self.concept_emb_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            emb_size,
                            emb_size,
                        ),
                        emb_act,
                    ])
                )
        for i in range(n_discovered_concepts):
            if self.cond_discovery:
                if self.shared_emb_generator:
                    if len(self.discovered_concept_emb_generators) == 0:
                        self.discovered_concept_emb_generators.append(torch.nn.Sequential(*[
                            torch.nn.Linear(
                                list(
                                    self.pre_concept_model.modules()
                                )[-1].out_features + (self.emb_size * self.n_concepts),
                                emb_size,
                            ),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(
                                emb_size,
                                emb_size,
                            ),
                            emb_act,
                        ]))
                    else:
                        self.discovered_concept_emb_generators.append(
                            self.discovered_concept_emb_generators[0]
                        )
                else:
                    self.discovered_concept_emb_generators.append(torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            emb_size,
                            emb_size,
                        ),
                        emb_act,
                    ]))
            else:
                if self.shared_emb_generator:
                    self.discovered_concept_emb_generators.append(
                        self.concept_emb_generators[0]
                    )
                else:
                    self.discovered_concept_emb_generators.append(torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            emb_size,
                            emb_size,
                        ),
                        emb_act,
                    ]))

        if c2y_model is None:
            # Else we construct it here directly
            units = [
                (self.n_concepts + self.n_discovered_concepts) * self.emb_size
            ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model

        self.sig = torch.nn.Sigmoid()

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights, reduction='none')
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights,
                reduction='none',
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.tau = tau
        self.use_concept_groups = use_concept_groups
        self._current_pred_concepts = None
        self._c_discovered_sem = None

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
        current_loss = 0.0
        if (self.intermediate_task_concept_loss != 0) and (
            self._current_pred_concepts is not None
        ):
            centers = []
            for i in range(self.n_discovered_concepts):
                # [Shape: (1, emb_size)]
                anchor_concept_pos_emb = torch.unsqueeze(
                    self.discovered_concept_embeddings[i, 0, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_pos_emb = torch.nn.functional.normalize(
                        anchor_concept_pos_emb,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_neg_emb = torch.unsqueeze(
                    self.discovered_concept_embeddings[i, 1, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_neg_emb = torch.nn.functional.normalize(
                        anchor_concept_neg_emb,
                        dim=0,
                    )
                center = (
                    0.5 * anchor_concept_neg_emb +
                    0.5 * anchor_concept_pos_emb
                )
                centers.append(center)
            centers = torch.cat(centers, dim=0)
            centers = torch.stack(
                [centers for _ in range(self._current_pred_concepts.shape[0])],
                dim=0,
            )
            new_input = torch.flatten(
                torch.cat(
                    [self._current_pred_concepts, centers],
                    dim=1,
                ),
                start_dim=1,
            )

            # CHANGES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # gt_concepts = []
            # for i in range(self.n_concepts):
            #     # [Shape: (1, emb_size)]
            #     anchor_concept_pos_emb = torch.unsqueeze(
            #         self.concept_embeddings[i, 0, :],
            #         dim=0,
            #     )
            #     if self.normalize_embs:
            #         anchor_concept_pos_emb = torch.nn.functional.normalize(
            #             anchor_concept_pos_emb,
            #             dim=0,
            #         )
            #     # [Shape: (1, emb_size)]
            #     anchor_concept_neg_emb = torch.unsqueeze(
            #         self.concept_embeddings[i, 1, :],
            #         dim=0,
            #     )
            #     if self.normalize_embs:
            #         anchor_concept_neg_emb = torch.nn.functional.normalize(
            #             anchor_concept_neg_emb,
            #             dim=0,
            #         )
            #     # [Shape: (B, 1, emb_size)]
            #     gt_concept = (
            #         (1 - c[:, i:i+1]) * anchor_concept_neg_emb +
            #         c[:, i:i+1] * anchor_concept_pos_emb
            #     ).unsqueeze(dim=1)
            #     gt_concepts.append(gt_concept)
            # gt_concepts = torch.cat(gt_concepts, dim=1)
            # new_input = torch.flatten(
            #     torch.cat(
            #         [gt_concepts, centers],
            #         dim=1,
            #     ),
            #     start_dim=1,
            # )
            # # END CHANGES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            intermediate_y_logits = self.c2y_model(
                new_input
            )
            current_loss = self.intermediate_task_concept_loss * self.loss_task(
                (
                    intermediate_y_logits if intermediate_y_logits.shape[-1] > 1
                    else intermediate_y_logits.reshape(-1)
                ),
                y,
            )
            current_loss = self._loss_mean(current_loss, y=y)
            # and reset it
            self._current_pred_concepts = None

        if (self.discovered_probs_entropy != 0) and (
            self._c_discovered_sem is not None
        ):
            current_loss += torch.mean(
                self.discovered_probs_entropy * _binary_entropy(
                    self._c_discovered_sem
                )
            )
            # and reset it
            self._c_discovered_sem = None

        if self.contrastive_loss_weight == 0 or (
            self.n_discovered_concepts == 0
        ):
            return current_loss

        extra_losses = []
        for concept_idx in range(self.n_concepts):
            for extra_concept_idx in range(self.n_discovered_concepts):
                # CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # extra_losses.append(self.contrastive_loss_fn(
                #     self.concept_embeddings[concept_idx, 0, :] - self.concept_embeddings[concept_idx, 1, :],
                #     self.discovered_concept_embeddings[
                #         extra_concept_idx,
                #         0,
                #         :
                #     ] - self.discovered_concept_embeddings[
                #         extra_concept_idx,
                #         1,
                #         :
                #     ],
                #     self.contrastive_target.to(
                #         self.discovered_concept_embeddings.device
                #     ),
                # ))
                # END OF CHANGE
                for selected_idx_1 in [0, 1]:
                    for selected_idx_2 in [0, 1]:
                        extra_losses.append(self.contrastive_loss_fn(
                            self.concept_embeddings[concept_idx, selected_idx_1, :],
                            self.discovered_concept_embeddings[
                                extra_concept_idx,
                                selected_idx_2,
                                :
                            ],
                            self.contrastive_target.to(
                                self.discovered_concept_embeddings.device
                            ),
                        ))

        total_loss = 0.0
        for loss in extra_losses:
            total_loss += loss
        return current_loss + self.contrastive_loss_weight * total_loss/len(extra_losses)

    def _relaxed_multi_bernoulli_sample(self, probs, temperature=1, idx=None):
        # Sample from a standard Gaussian first to perform the
        # reparameterization trick
        # shape = (probs.shape[0], self.L.shape[0], probs.shape[0])
        # epsilon = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)).to(
        #     probs.device
        # )
        # v = torch.transpose(torch.matmul(self.L.to(probs.device).unsqueeze(0), epsilon), -2, -1)
        # u = Gaussian_CDF(v)
        # if idx is not None:
        #     return torch.sigmoid(
        #         1.0/temperature * (
        #             log(probs) - log(1. - probs) + log(u[:, idx]) - log(1. - u[:, idx])
        #         )
        #     )


        shape = (probs.shape[0],)
        epsilon = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)).to(
            probs.device
        )
        u = Gaussian_CDF(epsilon)
        return torch.sigmoid(
            1.0/temperature * (
                log(probs) - log(1. - probs) + log(u) - log(1. - u)
            )
        )

        # epsilon = torch.rand(probs.shape[0]).to(
        #     probs.device
        # )
        # return torch.where(
        #     probs >= epsilon,
        #     torch.ones((probs.shape[0],)).to(probs.device),
        #     torch.zeros((probs.shape[0],)).to(probs.device),
        # )

        # return torch.where(
        #     probs >= 0.5,
        #     torch.ones((probs.shape[0],)).to(probs.device),
        #     torch.zeros((probs.shape[0],)).to(probs.device),
        # )

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        out_embeddings=None,
        pred_concepts=None,
        dynamic_mixtures=None,
        pre_c=None,
        dyn_c_true=None,
        dyn_prob=None,
        dyn_intervention_idxs=None,
        **kwargs,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if train and (self.dyn_training_intervention_prob != 0) and (
            (dyn_intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            dyn_intervention_idxs = torch.stack(
                [
                    torch.bernoulli(
                        self.dyn_ones * self.dyn_training_intervention_prob,
                    ) for _ in range(c_true.shape[0])
                ],
                dim=0
            )

            if dyn_c_true is None:
                # Then we randomly will set them
                dyn_c_true = torch.stack(
                [
                    torch.bernoulli(
                        self.dyn_ones * 0.5,
                    ) for _ in range(c_true.shape[0])
                ],
                dim=0
            )
        if (c_true is None) or (
            (intervention_idxs is None) and
            (dyn_intervention_idxs is None)
        ):
            if dynamic_mixtures is None:
                bottleneck = torch.flatten(
                    pred_concepts,
                    start_dim=1,
                )
            else:
                bottleneck = torch.flatten(
                    torch.cat(
                        [pred_concepts, dynamic_mixtures],
                        dim=1,
                    ),
                    start_dim=1,
                )
            return prob, intervention_idxs, bottleneck
        if intervention_idxs is None:
            intervention_idxs = torch.zeros((c_true.shape[0], self.n_concepts))
        if dyn_intervention_idxs is None:
            dyn_intervention_idxs = torch.zeros((c_true.shape[0], self.n_discovered_concepts))

        # First mixed trained concepts
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        output = prob * (1 - intervention_idxs) + intervention_idxs * c_true

        # [Shape: (1, n_concepts, emb_size)]
        pos_anchors = self.concept_embeddings[:, 0, :].unsqueeze(0)
        if self.normalize_embs:
            pos_anchors = torch.nn.functional.normalize(
                pos_anchors,
                dim=1,
            )
        # [Shape: (1, n_concepts, emb_size)]
        neg_anchors = self.concept_embeddings[:, 1, :].unsqueeze(0)
        if self.normalize_embs:
            neg_anchors = torch.nn.functional.normalize(
                neg_anchors,
                dim=1,
            )
        # [Shape: (B, n_concepts, 1)]
        extended_intervention_idxs = torch.unsqueeze(
            intervention_idxs,
            dim=-1,
        )
        # [Shape: (B, n_concepts, 1)]
        extended_c_true = c_true.unsqueeze(-1)
        # [Shape: (B, n_concepts, emb_size)]
        ground_truth_anchors = pos_anchors * extended_c_true + (1 - extended_c_true) * neg_anchors
        # [Shape: (B, n_concepts, emb_size)]
        pred_concepts = (
            (1 - extended_intervention_idxs) * pred_concepts +
            extended_intervention_idxs * ground_truth_anchors
        )
        if train:
            self._current_pred_concepts = pred_concepts


        # Next mixed discovered concepts
        if (dynamic_mixtures is not None) and (
            (dyn_c_true is not None) and
            (dyn_intervention_idxs is not None)
        ):

            dyn_intervention_idxs = dyn_intervention_idxs.type(torch.FloatTensor)
            dyn_intervention_idxs = dyn_intervention_idxs.to(prob.device)

            # [Shape: (1, n_concepts, emb_size)]
            pos_anchors = self.discovered_concept_embeddings[:, 0, :].unsqueeze(0)
            if self.normalize_embs:
                pos_anchors = torch.nn.functional.normalize(
                    pos_anchors,
                    dim=1,
                )
            # [Shape: (1, n_concepts, emb_size)]
            neg_anchors = self.discovered_concept_embeddings[:, 1, :].unsqueeze(0)
            if self.normalize_embs:
                neg_anchors = torch.nn.functional.normalize(
                    neg_anchors,
                    dim=1,
                )
            # [Shape: (B, n_concepts, 1)]
            extended_dyn_intervention_idxs = torch.unsqueeze(
                dyn_intervention_idxs,
                dim=-1,
            )
            # [Shape: (B, n_concepts, 1)]
            extended_dyn_c_true = dyn_c_true.unsqueeze(-1).to(pos_anchors.device)
            # [Shape: (B, n_concepts, emb_size)]
            ground_truth_anchors = pos_anchors * extended_dyn_c_true + (1 - extended_dyn_c_true) * neg_anchors
            # ground_truth_anchors = (pos_anchors + neg_anchors)/2 # pos_anchors * extended_dyn_c_true + (1 - extended_dyn_c_true) * neg_anchors
            # [Shape: (B, n_concepts, emb_size)]
            dynamic_mixtures = (
                (1 - extended_dyn_intervention_idxs) * dynamic_mixtures +
                extended_dyn_intervention_idxs * ground_truth_anchors
            )

        # Then time to mix!
        if dynamic_mixtures is None:
            bottleneck = torch.flatten(
                pred_concepts,
                start_dim=1,
            )
        else:
            # [Shape: (B, n_concepts, 2*emb_size)]
            if self.cond_discovery:
                new_dynamic_mixtures = []
                used_pre_c = torch.cat(
                    [pre_c, torch.flatten(pred_concepts, start_dim=1)],
                    dim=-1,
                )
                for i, discovered_concept_emb_generator in enumerate(
                    self.discovered_concept_emb_generators
                ):
                    # [Shape: (B, emb_size)]
                    pred_discovered_concept_embs = discovered_concept_emb_generator(
                        used_pre_c
                    )
                    if self.normalize_embs:
                        pred_discovered_concept_embs = torch.nn.functional.normalize(
                            pred_discovered_concept_embs,
                            dim=0,
                        )
                    # [Shape: (1, emb_size)]
                    anchor_concept_pos_emb = torch.unsqueeze(
                        self.discovered_concept_embeddings[i, 0, :],
                        dim=0,
                    )
                    if self.normalize_embs:
                        anchor_concept_pos_emb = torch.nn.functional.normalize(
                            anchor_concept_pos_emb,
                            dim=0,
                        )
                    # [Shape: (1, emb_size)]
                    anchor_concept_neg_emb = torch.unsqueeze(
                        self.discovered_concept_embeddings[i, 1, :],
                        dim=0,
                    )
                    if self.normalize_embs:
                        anchor_concept_neg_emb = torch.nn.functional.normalize(
                            anchor_concept_neg_emb,
                            dim=0,
                        )
                    # [Shape: (B)]
                    prob = self.sig(
                        self.discovered_contrastive_scale[i] * (
                            (anchor_concept_neg_emb - pred_discovered_concept_embs).pow(2).sum(-1).sqrt() -
                            (anchor_concept_pos_emb - pred_discovered_concept_embs).pow(2).sum(-1).sqrt()
                        )
                    )
                    prob = torch.unsqueeze(prob, dim=-1)
                    if self.mix_ground_truth_embs:
                        pred_discovered_concept_embs = prob * anchor_concept_pos_emb + (1 - prob) * anchor_concept_neg_emb

                    new_dynamic_mixtures.append(
                        torch.unsqueeze(pred_discovered_concept_embs, dim=1)
                    )
                dynamic_mixtures = torch.cat(new_dynamic_mixtures, dim=1)
            bottleneck = torch.flatten(
                torch.cat(
                    [pred_concepts, dynamic_mixtures],
                    dim=1,
                ),
                start_dim=1,
            )
        return output, intervention_idxs, bottleneck

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            c_sem = []
            c_discovered_sem = []
            dynamic_mixtures = []
            pred_concepts = []

            # First predict all the concept probabilities
            for i, concept_emb_generator in enumerate(
                self.concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                pred_concept_embeddings = concept_emb_generator(pre_c)
                if self.normalize_embs:
                    pred_concept_embeddings = torch.nn.functional.normalize(
                        pred_concept_embeddings,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_pos_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 0, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_pos_emb = torch.nn.functional.normalize(
                        anchor_concept_pos_emb,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_neg_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 1, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_neg_emb = torch.nn.functional.normalize(
                        anchor_concept_neg_emb,
                        dim=0,
                    )
                # [Shape: (B)]
                prob = self.sig(
                    self.contrastive_scale[i] * (
                        (anchor_concept_neg_emb - pred_concept_embeddings).pow(2).sum(-1).sqrt() -
                        (anchor_concept_pos_emb - pred_concept_embeddings).pow(2).sum(-1).sqrt()
                    )
                )
                # [Shape: (B, 1)]
                if self.sample_probs:
                    if training:
                        prob = self._relaxed_multi_bernoulli_sample(prob, idx=i)
                    else:
                        prob = (prob >= 0.5).type(prob.type())
                prob = torch.unsqueeze(prob, dim=-1)
                if self.mix_ground_truth_embs:
                    pred_concept_embeddings = prob * anchor_concept_pos_emb + (1 - prob) * anchor_concept_neg_emb

                c_sem.append(prob)
                pred_concepts.append(
                    torch.unsqueeze(pred_concept_embeddings, dim=1)
                )
            c_sem = torch.cat(c_sem, dim=-1)
            pred_concepts = torch.cat(pred_concepts, dim=1)
            if self.cond_discovery:
                used_pre_c = torch.cat(
                    [pre_c, torch.flatten(pred_concepts, start_dim=1)],
                    dim=-1,
                )
            else:
                used_pre_c = pre_c

            for i, discovered_concept_emb_generator in enumerate(
                 self.discovered_concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                pred_discovered_concept_embs = discovered_concept_emb_generator(
                    used_pre_c
                )
                if self.normalize_embs:
                    pred_discovered_concept_embs = torch.nn.functional.normalize(
                        pred_discovered_concept_embs,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_pos_emb = torch.unsqueeze(
                    self.discovered_concept_embeddings[i, 0, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_pos_emb = torch.nn.functional.normalize(
                        anchor_concept_pos_emb,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_neg_emb = torch.unsqueeze(
                    self.discovered_concept_embeddings[i, 1, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_neg_emb = torch.nn.functional.normalize(
                        anchor_concept_neg_emb,
                        dim=0,
                    )
                # [Shape: (B)]
                prob = self.sig(
                    self.discovered_contrastive_scale[i] * (
                        (anchor_concept_neg_emb - pred_discovered_concept_embs).pow(2).sum(-1).sqrt() -
                        (anchor_concept_pos_emb - pred_discovered_concept_embs).pow(2).sum(-1).sqrt()
                    )
                )
                # [Shape: (B, 1)]
                # if self.sample_probs:
                #     if training:
                #         prob = self._relaxed_multi_bernoulli_sample(prob, idx=i)
                #     else:
                #         prob = (prob >= 0.5).type(prob.type())
                prob = torch.unsqueeze(prob, dim=-1)
                if self.mix_ground_truth_embs:
                    pred_discovered_concept_embs = prob * anchor_concept_pos_emb + (1 - prob) * anchor_concept_neg_emb
                c_discovered_sem.append(prob)

                dynamic_mixtures.append(
                    torch.unsqueeze(pred_discovered_concept_embs, dim=1)
                )

            if self.n_discovered_concepts != 0:
                c_discovered_sem = torch.cat(c_discovered_sem, dim=-1)
                dynamic_mixtures = torch.cat(dynamic_mixtures, dim=1)
            else:
                dynamic_mixtures = None
                c_discovered_sem = None
            latent = c_sem, pred_concepts, dynamic_mixtures, c_discovered_sem
        else:
            c_sem, pred_concepts, dynamic_mixtures, c_discovered_sem = latent
        if training:
            self._current_pred_concepts = pred_concepts
            self._c_discovered_sem = c_discovered_sem
        return c_sem, None, None, {
            "pred_concepts": pred_concepts,
            "dynamic_mixtures": dynamic_mixtures,
            "pre_c": pre_c,
            "latent": (dynamic_mixtures, c_discovered_sem),
        }

    def _loss_mean(self, losses, y, loss_weights=None):
        if loss_weights is None:
            loss_weights = torch.ones_like(y)
        if self.loss_task.weight is not None:
            norm_constant = torch.gather(self.loss_task.weight, 0, y)
            return torch.sum(losses, dim=-1) / torch.sum(
                loss_weights * norm_constant,
                dim=-1,
            )
        return torch.sum(losses, dim=-1) / torch.sum(loss_weights, dim=-1)

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

        int_probs = []
        og_int_probs = self.training_intervention_prob
        if isinstance(self.training_intervention_prob, list) and train:
            int_probs = self.training_intervention_prob
        elif train:
            int_probs = [self.training_intervention_prob]
        else:
            int_probs = [0]

        task_loss = 0.0
        loss_weights = 0.0
        c_sem = None
        c_logits = None
        for int_prob in int_probs:
            self.training_intervention_prob = int_prob
            outputs = self._forward(
                x,
                intervention_idxs=intervention_idxs,
                c=c,
                y=y,
                train=train,
                competencies=competencies,
                prev_interventions=prev_interventions,
                output_interventions=True,
            )
            if c_sem is None:
                c_sem, c_logits = outputs[0], outputs[1]

            y_logits = outputs[2]
            int_mask = outputs[3]
            current_task_loss = self.task_loss_weight * self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            if int_mask is not None:
                scaling = torch.pow(
                    self.intervention_task_discount,
                    torch.sum(int_mask, dim=-1),
                )
                loss_weights += scaling
                task_loss += current_task_loss * scaling
            else:
                task_loss += current_task_loss
                loss_weights += torch.ones_like(current_task_loss)
        task_loss = self._loss_mean(
            task_loss,
            y=y,
            loss_weights=loss_weights,
        )
        if not isinstance(task_loss, (float, int)):
            task_loss_scalar = task_loss.detach()
        else:
            task_loss_scalar = task_loss
        self.training_intervention_prob = og_int_probs

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





class ResidualMixingConceptEmbeddingModel(ConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        n_discovered_concepts=4,
        training_intervention_prob=0.25,
        dyn_training_intervention_prob=0,
        embedding_activation="leakyrelu",
        concept_loss_weight=1,
        task_loss_weight=1,
        contrastive_loss_weight=0,
        intermediate_task_concept_loss=0,
        intervention_task_discount=1,
        discovered_probs_entropy=0,

        mix_ground_truth_embs=True,
        shared_emb_generator=False,
        normalize_embs=False,
        sample_probs=False,
        cov_mat=None,
        cond_discovery=False,

        c2y_model=None,
        c2y_layers=None,
        residual_c2y_model=None,
        residual_c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        tau=1,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.dyn_training_intervention_prob = dyn_training_intervention_prob
        self.output_latent = output_latent
        self.mix_ground_truth_embs = mix_ground_truth_embs
        self.normalize_embs = normalize_embs
        self.sample_probs = sample_probs
        self.intervention_task_discount = intervention_task_discount
        self.intermediate_task_concept_loss = intermediate_task_concept_loss
        self.discovered_probs_entropy = discovered_probs_entropy
        if cov_mat is None:
            cov_mat = np.eye(n_concepts, dtype=np.float32)
        self.cov_mat = cov_mat
        self.L = torch.tensor(
            scipy.linalg.cholesky(self.cov_mat, lower=True).astype(
                np.float32
            )
        )

        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)
        if self.dyn_training_intervention_prob != 0:
            self.dyn_ones = torch.ones(n_discovered_concepts)

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
        self.contrastive_loss_weight = contrastive_loss_weight
        if self.contrastive_loss_weight != 0:
            self.contrastive_target = -torch.ones([])
        self.contrastive_loss_fn = torch.nn.CosineEmbeddingLoss(
            margin=0.0,
            size_average=None,
            reduce=None,
            reduction='mean',
        )
        # self.contrastive_loss_fn = lambda x1, x2, *args, **kwargs: (x1 - x2).pow(2).sum(-1)
        self.top_k_accuracy = top_k_accuracy

        self.emb_size = emb_size
        self.concept_embeddings = torch.nn.Parameter(
            torch.rand(self.n_concepts, 2, self.emb_size),
        )
        self.n_discovered_concepts = n_discovered_concepts
        if self.n_discovered_concepts != 0:
            self.discovered_concept_embeddings = torch.nn.Parameter(torch.rand(
                self.n_discovered_concepts,
                2,
                self.emb_size,
            ))
            self.discovered_contrastive_scale = torch.nn.Parameter(
                torch.rand((self.n_discovered_concepts,))
            )
        self.contrastive_scale = torch.nn.Parameter(
            torch.rand((self.n_concepts,))
        )

        self.concept_emb_generators = torch.nn.ModuleList()
        self.discovered_concept_emb_generators = torch.nn.ModuleList()
        self.shared_emb_generator = shared_emb_generator
        self.cond_discovery = cond_discovery
        for i in range(n_concepts):
            if embedding_activation is None:
                emb_act = torch.nn.Identity()
            elif embedding_activation == "sigmoid":
                emb_act = torch.nn.Sigmoid()
            elif embedding_activation == "leakyrelu":
                emb_act = torch.nn.LeakyReLU()
            elif embedding_activation == "relu":
                emb_act = torch.nn.ReLU()
            else:
                raise ValueError(
                    f'Unsupported embedding activation "{embedding_activation}"'
                )
            if self.shared_emb_generator:
                if len(self.concept_emb_generators) == 0:
                    self.concept_emb_generators.append(
                        torch.nn.Sequential(*[
                            torch.nn.Linear(
                                list(
                                    self.pre_concept_model.modules()
                                )[-1].out_features,
                                emb_size,
                            ),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(
                                emb_size,
                                emb_size,
                            ),
                            emb_act,
                        ])
                    )
                else:
                    self.concept_emb_generators.append(
                        self.concept_emb_generators[0]
                    )
            else:
                self.concept_emb_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            emb_size,
                            emb_size,
                        ),
                        emb_act,
                    ])
                )
        for i in range(n_discovered_concepts):
            if self.cond_discovery:
                self.discovered_concept_emb_generators.append(torch.nn.Sequential(*[
                    torch.nn.Linear(
                        list(
                            self.pre_concept_model.modules()
                        )[-1].out_features,
                        emb_size,
                    ),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(
                        emb_size,
                        emb_size,
                    ),
                    emb_act,
                ]))
            else:
                if self.shared_emb_generator:
                    self.discovered_concept_emb_generators.append(
                        self.concept_emb_generators[0]
                    )
                else:
                    self.discovered_concept_emb_generators.append(torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(
                            emb_size,
                            emb_size,
                        ),
                        emb_act,
                    ]))

        if c2y_model is None:
            # Else we construct it here directly
            units = [
                self.n_concepts * self.emb_size
            ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model


        if residual_c2y_model is None:
            # Else we construct it here directly
            units = [
                (self.n_concepts + self.n_discovered_concepts) * self.emb_size + n_tasks
            ] + (residual_c2y_layers or c2y_layers) + [n_tasks]
            residual_layers = []
            for i in range(1, len(units)):
                residual_layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    residual_layers.append(torch.nn.LeakyReLU())
            self.residual_c2y_model = torch.nn.Sequential(*residual_layers)
        else:
            self.residual_c2y_model = residual_c2y_model

        self.sig = torch.nn.Sigmoid()

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights, reduction='none')
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights,
                reduction='none',
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.tau = tau
        self.use_concept_groups = use_concept_groups
        self._current_pred_concepts = None
        self._c_discovered_sem = None
        self._y_pred = None
        self._residual_y_pred = None

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
        current_loss = 0.0
        if (self.discovered_probs_entropy != 0) and (
            self._c_discovered_sem is not None
        ):
            current_loss += torch.mean(
                self.discovered_probs_entropy * _binary_entropy(
                    self._c_discovered_sem
                )
            )
            # and reset it
            self._c_discovered_sem = None

        if (self._y_pred is not None):
            current_loss += self.task_loss_weight * self._loss_mean(
                self.loss_task(self._y_pred, y),
                y=y,
            )

            current_loss += torch.mean(torch.abs(self._residual_y_pred))
            # and reset it
            self._y_pred = None
            self._residual_y_pred = None

        if self.contrastive_loss_weight == 0 or (
            self.n_discovered_concepts == 0
        ):
            return current_loss

        extra_losses = []
        for concept_idx in range(self.n_concepts):
            for extra_concept_idx in range(self.n_discovered_concepts):
                for selected_idx_1 in [0, 1]:
                    for selected_idx_2 in [0, 1]:
                        extra_losses.append(self.contrastive_loss_fn(
                            self.concept_embeddings[concept_idx, selected_idx_1, :],
                            self.discovered_concept_embeddings[
                                extra_concept_idx,
                                selected_idx_2,
                                :
                            ],
                            self.contrastive_target.to(
                                self.discovered_concept_embeddings.device
                            ),
                        ))

        total_loss = 0.0
        for loss in extra_losses:
            total_loss += loss
        return current_loss + self.contrastive_loss_weight * total_loss/len(extra_losses)

    def _relaxed_multi_bernoulli_sample(self, probs, temperature=1, idx=None):
        shape = (probs.shape[0],)
        epsilon = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)).to(
            probs.device
        )
        u = Gaussian_CDF(epsilon)
        return torch.sigmoid(
            1.0/temperature * (
                log(probs) - log(1. - probs) + log(u) - log(1. - u)
            )
        )

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        out_embeddings=None,
        pred_concepts=None,
        dynamic_mixtures=None,
        pre_c=None,
        dyn_c_true=None,
        dyn_prob=None,
        dyn_intervention_idxs=None,
        **kwargs,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if train and (self.dyn_training_intervention_prob != 0) and (
            (dyn_intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            dyn_intervention_idxs = torch.stack(
                [
                    torch.bernoulli(
                        self.dyn_ones * self.dyn_training_intervention_prob,
                    ).unsqueeze(-1) for _ in range(c_true.shape[0])
                ],
                dim=-1
            )

            if dyn_c_true is None:
                # Then we randomly will set them
                dyn_c_true = torch.stack(
                [
                    (torch.bernoulli(
                        self.dyn_ones * 0.5,
                    ).unsqueeze(-1)) for _ in range(c_true.shape[0])
                ],
                dim=-1
            )
        if (c_true is None) or (
            (intervention_idxs is None) and
            (dyn_intervention_idxs is None)
        ):
            if dynamic_mixtures is None:
                bottleneck = torch.flatten(
                    pred_concepts,
                    start_dim=1,
                )
            else:
                bottleneck = torch.cat(
                    [pred_concepts, dynamic_mixtures],
                    dim=1,
                )
            return prob, intervention_idxs, bottleneck
        if intervention_idxs is None:
            intervention_idxs = torch.zeros((c_true.shape[0], self.n_concepts))
        if dyn_intervention_idxs is None:
            dyn_intervention_idxs = torch.zeros((c_true.shape[0], self.n_discovered_concepts))

        # First mixed trained concepts
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        output = prob * (1 - intervention_idxs) + intervention_idxs * c_true

        # [Shape: (1, n_concepts, emb_size)]
        pos_anchors = self.concept_embeddings[:, 0, :].unsqueeze(0)
        if self.normalize_embs:
            pos_anchors = torch.nn.functional.normalize(
                pos_anchors,
                dim=1,
            )
        # [Shape: (1, n_concepts, emb_size)]
        neg_anchors = self.concept_embeddings[:, 1, :].unsqueeze(0)
        if self.normalize_embs:
            neg_anchors = torch.nn.functional.normalize(
                neg_anchors,
                dim=1,
            )
        # [Shape: (B, n_concepts, 1)]
        extended_intervention_idxs = torch.unsqueeze(
            intervention_idxs,
            dim=-1,
        )
        # [Shape: (B, n_concepts, 1)]
        extended_c_true = c_true.unsqueeze(-1)
        # [Shape: (B, n_concepts, emb_size)]
        ground_truth_anchors = pos_anchors * extended_c_true + (1 - extended_c_true) * neg_anchors
        # [Shape: (B, n_concepts, emb_size)]
        pred_concepts = (
            (1 - extended_intervention_idxs) * pred_concepts +
            extended_intervention_idxs * ground_truth_anchors
        )
        if train:
            self._current_pred_concepts = pred_concepts


        # Next mixed discovered concepts
        if (dynamic_mixtures is not None) and (
            (dyn_prob is not None) and
            (dyn_c_true is not None)
        ):
            dyn_intervention_idxs = dyn_intervention_idxs.type(torch.FloatTensor)
            dyn_intervention_idxs = dyn_intervention_idxs.to(prob.device)

            # [Shape: (1, n_concepts, emb_size)]
            pos_anchors = self.discovered_concept_embeddings[:, 0, :].unsqueeze(0)
            if self.normalize_embs:
                pos_anchors = torch.nn.functional.normalize(
                    pos_anchors,
                    dim=1,
                )
            # [Shape: (1, n_concepts, emb_size)]
            neg_anchors = self.discovered_concept_embeddings[:, 1, :].unsqueeze(0)
            if self.normalize_embs:
                neg_anchors = torch.nn.functional.normalize(
                    neg_anchors,
                    dim=1,
                )
            # [Shape: (B, n_concepts, 1)]
            extended_dyn_intervention_idxs = torch.unsqueeze(
                dyn_intervention_idxs,
                dim=-1,
            )
            # [Shape: (B, n_concepts, 1)]
            extended_dyn_c_true = dyn_c_true.unsqueeze(-1)
            # [Shape: (B, n_concepts, emb_size)]
            ground_truth_anchors = (pos_anchors + neg_anchors)/2 # pos_anchors * extended_dyn_c_true + (1 - extended_dyn_c_true) * neg_anchors
            # [Shape: (B, n_concepts, emb_size)]
            dynamic_mixtures = (
                (1 - extended_dyn_intervention_idxs) * dynamic_mixtures +
                extended_dyn_intervention_idxs * ground_truth_anchors
            )

        # Then time to mix!
        if dynamic_mixtures is None:
            bottleneck = pred_concepts
        else:
            # [Shape: (B, n_concepts, 2*emb_size)]
            if self.cond_discovery:
                new_dynamic_mixtures = []
                used_pre_c = torch.cat(
                    [pre_c, torch.flatten(pred_concepts, start_dim=1)],
                    dim=-1,
                )
                for i, discovered_concept_emb_generator in enumerate(
                    self.discovered_concept_emb_generators
                ):
                    # [Shape: (B, emb_size)]
                    pred_discovered_concept_embs = discovered_concept_emb_generator(
                        used_pre_c
                    )
                    if self.normalize_embs:
                        pred_discovered_concept_embs = torch.nn.functional.normalize(
                            pred_discovered_concept_embs,
                            dim=0,
                        )
                    # [Shape: (1, emb_size)]
                    anchor_concept_pos_emb = torch.unsqueeze(
                        self.discovered_concept_embeddings[i, 0, :],
                        dim=0,
                    )
                    if self.normalize_embs:
                        anchor_concept_pos_emb = torch.nn.functional.normalize(
                            anchor_concept_pos_emb,
                            dim=0,
                        )
                    # [Shape: (1, emb_size)]
                    anchor_concept_neg_emb = torch.unsqueeze(
                        self.discovered_concept_embeddings[i, 1, :],
                        dim=0,
                    )
                    if self.normalize_embs:
                        anchor_concept_neg_emb = torch.nn.functional.normalize(
                            anchor_concept_neg_emb,
                            dim=0,
                        )
                    # [Shape: (B)]
                    prob = self.sig(
                        self.discovered_contrastive_scale[i] * (
                            (anchor_concept_neg_emb - pred_discovered_concept_embs).pow(2).sum(-1).sqrt() -
                            (anchor_concept_pos_emb - pred_discovered_concept_embs).pow(2).sum(-1).sqrt()
                        )
                    )
                    prob = torch.unsqueeze(prob, dim=-1)
                    if self.mix_ground_truth_embs:
                        pred_discovered_concept_embs = prob * anchor_concept_pos_emb + (1 - prob) * anchor_concept_neg_emb

                    new_dynamic_mixtures.append(
                        torch.unsqueeze(pred_discovered_concept_embs, dim=1)
                    )
                dynamic_mixtures = torch.cat(new_dynamic_mixtures, dim=1)
            bottleneck = torch.cat(
                [pred_concepts, dynamic_mixtures],
                dim=1,
            )
        return output, intervention_idxs, bottleneck

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            c_sem = []
            c_discovered_sem = []
            dynamic_mixtures = []
            pred_concepts = []

            # First predict all the concept probabilities
            for i, concept_emb_generator in enumerate(
                self.concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                pred_concept_embeddings = concept_emb_generator(pre_c)
                if self.normalize_embs:
                    pred_concept_embeddings = torch.nn.functional.normalize(
                        pred_concept_embeddings,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_pos_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 0, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_pos_emb = torch.nn.functional.normalize(
                        anchor_concept_pos_emb,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_neg_emb = torch.unsqueeze(
                    self.concept_embeddings[i, 1, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_neg_emb = torch.nn.functional.normalize(
                        anchor_concept_neg_emb,
                        dim=0,
                    )
                # [Shape: (B)]
                prob = self.sig(
                    self.contrastive_scale[i] * (
                        (anchor_concept_neg_emb - pred_concept_embeddings).pow(2).sum(-1).sqrt() -
                        (anchor_concept_pos_emb - pred_concept_embeddings).pow(2).sum(-1).sqrt()
                    )
                )
                # [Shape: (B, 1)]
                if self.sample_probs:
                    if training:
                        prob = self._relaxed_multi_bernoulli_sample(prob, idx=i)
                    else:
                        prob = (prob >= 0.5).type(prob.type())
                prob = torch.unsqueeze(prob, dim=-1)
                if self.mix_ground_truth_embs:
                    pred_concept_embeddings = prob * anchor_concept_pos_emb + (1 - prob) * anchor_concept_neg_emb

                c_sem.append(prob)
                pred_concepts.append(
                    torch.unsqueeze(pred_concept_embeddings, dim=1)
                )
            c_sem = torch.cat(c_sem, dim=-1)
            pred_concepts = torch.cat(pred_concepts, dim=1)
            if self.cond_discovery:
                used_pre_c = torch.cat(
                    [pre_c, torch.flatten(pred_concepts, start_dim=1)],
                    dim=-1,
                )
            else:
                used_pre_c = pre_c

            for i, discovered_concept_emb_generator in enumerate(
                 self.discovered_concept_emb_generators
            ):
                # [Shape: (B, emb_size)]
                pred_discovered_concept_embs = discovered_concept_emb_generator(
                    used_pre_c
                )
                if self.normalize_embs:
                    pred_discovered_concept_embs = torch.nn.functional.normalize(
                        pred_discovered_concept_embs,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_pos_emb = torch.unsqueeze(
                    self.discovered_concept_embeddings[i, 0, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_pos_emb = torch.nn.functional.normalize(
                        anchor_concept_pos_emb,
                        dim=0,
                    )
                # [Shape: (1, emb_size)]
                anchor_concept_neg_emb = torch.unsqueeze(
                    self.discovered_concept_embeddings[i, 1, :],
                    dim=0,
                )
                if self.normalize_embs:
                    anchor_concept_neg_emb = torch.nn.functional.normalize(
                        anchor_concept_neg_emb,
                        dim=0,
                    )
                # [Shape: (B)]
                prob = self.sig(
                    self.discovered_contrastive_scale[i] * (
                        (anchor_concept_neg_emb - pred_discovered_concept_embs).pow(2).sum(-1).sqrt() -
                        (anchor_concept_pos_emb - pred_discovered_concept_embs).pow(2).sum(-1).sqrt()
                    )
                )
                # [Shape: (B, 1)]
                # if self.sample_probs:
                #     if training:
                #         prob = self._relaxed_multi_bernoulli_sample(prob, idx=i)
                #     else:
                #         prob = (prob >= 0.5).type(prob.type())
                prob = torch.unsqueeze(prob, dim=-1)
                if self.mix_ground_truth_embs:
                    pred_discovered_concept_embs = prob * anchor_concept_pos_emb + (1 - prob) * anchor_concept_neg_emb
                c_discovered_sem.append(prob)

                dynamic_mixtures.append(
                    torch.unsqueeze(pred_discovered_concept_embs, dim=1)
                )

            if self.n_discovered_concepts != 0:
                c_discovered_sem = torch.cat(c_discovered_sem, dim=-1)
                dynamic_mixtures = torch.cat(dynamic_mixtures, dim=1)
            else:
                dynamic_mixtures = None
                c_discovered_sem = None
            latent = c_sem, pred_concepts, dynamic_mixtures, c_discovered_sem
        else:
            c_sem, pred_concepts, dynamic_mixtures, c_discovered_sem = latent
        if training:
            self._current_pred_concepts = pred_concepts
            self._c_discovered_sem = c_discovered_sem
        return c_sem, None, None, {
            "pred_concepts": pred_concepts,
            "dynamic_mixtures": dynamic_mixtures,
            "pre_c": pre_c,
            "latent": (dynamic_mixtures, c_discovered_sem),
        }

    def _loss_mean(self, losses, y, loss_weights=None):
        if loss_weights is None:
            loss_weights = torch.ones_like(y)
        if self.loss_task.weight is not None:
            norm_constant = torch.gather(self.loss_task.weight, 0, y)
            return torch.sum(losses, dim=-1) / torch.sum(
                loss_weights * norm_constant,
                dim=-1,
            )
        return torch.sum(losses, dim=-1) / torch.sum(loss_weights, dim=-1)

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
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

        c_sem, pos_embs, neg_embs, out_kwargs = self._generate_concept_embeddings(
            x=x,
            latent=latent,
            training=train,
        )

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=pos_embs,
                neg_embeddings=neg_embs,
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
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
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        probs, intervention_idxs, bottleneck = self._after_interventions(
            c_sem,
            pos_embeddings=pos_embs,
            neg_embeddings=neg_embs,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
            **out_kwargs
        )
        concept_mixtures = bottleneck[:, :self.n_concepts, :]
        if len(concept_mixtures.shape) > 2:
            concept_mixtures = concept_mixtures.view((concept_mixtures.shape[0], -1))
        residual_mixtures = bottleneck
        if len(residual_mixtures.shape) > 2:
            residual_mixtures = residual_mixtures.view((residual_mixtures.shape[0], -1))
        y_pred = self.c2y_model(concept_mixtures)
        residual_y_pred = self.residual_c2y_model(torch.cat([residual_mixtures, y_pred], dim=-1))
        if train:
            self._y_pred = y_pred
            self._residual_y_pred = residual_y_pred
        y_pred = y_pred + residual_y_pred
        tail_results = []
        if output_interventions:
            if (
                (intervention_idxs is not None) and
                isinstance(intervention_idxs, np.ndarray)
            ):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            if "latent" in out_kwargs:
                latent = (latent or tuple([])) + out_kwargs['latent']
            tail_results.append(latent)
        if output_embeddings and (not pos_embs is None) and (
            not neg_embs is None
        ):
            tail_results.append(pos_embs)
            tail_results.append(neg_embs)
        return tuple([c_sem, bottleneck, y_pred] + tail_results)

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

        int_probs = []
        og_int_probs = self.training_intervention_prob
        if isinstance(self.training_intervention_prob, list) and train:
            int_probs = self.training_intervention_prob
        elif train:
            int_probs = [self.training_intervention_prob]
        else:
            int_probs = [0]

        task_loss = 0.0
        loss_weights = 0.0
        c_sem = None
        c_logits = None
        for int_prob in int_probs:
            self.training_intervention_prob = int_prob
            outputs = self._forward(
                x,
                intervention_idxs=intervention_idxs,
                c=c,
                y=y,
                train=train,
                competencies=competencies,
                prev_interventions=prev_interventions,
                output_interventions=True,
            )
            if c_sem is None:
                c_sem, c_logits = outputs[0], outputs[1]

            y_logits = outputs[2]
            int_mask = outputs[3]
            current_task_loss = self.task_loss_weight * self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            if int_mask is not None:
                scaling = torch.pow(
                    self.intervention_task_discount,
                    torch.sum(int_mask, dim=-1),
                )
                loss_weights += scaling
                task_loss += current_task_loss * scaling
            else:
                task_loss += self._loss_mean(
                    current_task_loss,
                    y=y,
                )
                loss_weights += 1
        task_loss = self._loss_mean(
            task_loss,
            y=y,
            loss_weights=loss_weights,
        )
        if not isinstance(task_loss, (float, int)):
            task_loss_scalar = task_loss.detach()
        else:
            task_loss_scalar = task_loss
        self.training_intervention_prob = og_int_probs

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