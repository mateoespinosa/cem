import numpy as np
import pytorch_lightning as pl
import torch

from torchvision.models import resnet50

from cem.models.cbm import ConceptBottleneckModel
import cem.train.utils as utils



################################################################################
## Concept Embedding Models
################################################################################


class ConceptEmbeddingModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
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

        # New changes
        sim_penalty=0.0,
        l2_penalty=0.0,
        emb_pred_loss=0.0,
        prob_training_thresholding=0,
        prior_loss_term=0.0,
        cbm_mode=False,
        ##################### end ##########################

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        context_gen_out_size=None,

        top_k_accuracy=None,
    ):
        """
        Constructs a Concept Embedding Model (CEM) as defined by
        Espinosa Zarlenga et al. 2022.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CEM.
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
            CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the CEM).
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
            Module that maps this CEM's inputs to the latent space of the
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
        self.cbm_mode = cbm_mode
        if self.cbm_mode:
            emb_size = 1
            embedding_activation = None
        else:
            context_gen_out_size = context_gen_out_size or (2 * emb_size)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.sim_penalty = sim_penalty
        self.l2_penalty = l2_penalty

        self.emb_pred_loss = emb_pred_loss
        self._neg_embs = None
        self._pos_embs = None
        if self.emb_pred_loss > 0.0:
            # Then we will instantiate a linear layer that predicts the concept
            # label from the positive or negartive embeddings
            self.emb_pred_layer = torch.nn.Linear(
                emb_size,
                1
            )
        self.prior_loss_term = prior_loss_term
        self.prob_training_thresholding = prob_training_thresholding
        self._intervention_idxs = None
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
        for i in range(n_concepts):
            if embedding_activation is None:
                act_to_use = []
            elif embedding_activation == "sigmoid":
                act_to_use = [torch.nn.Sigmoid()]
            elif embedding_activation == "leakyrelu":
                act_to_use = [torch.nn.LeakyReLU()]
            elif embedding_activation == "relu":
                act_to_use = [torch.nn.ReLU()]
            else:
                raise ValueError(
                    f'Unsupported embedding activation "{embedding_activation}"'
                )
            if self.cbm_mode:
                # We simply output a single value per concept representing the
                # logit of the concept
                self.concept_context_generators.append(
                    torch.nn.Linear(
                        list(
                            self.pre_concept_model.modules()
                        )[-1].out_features,
                        1,
                    )
                )
                # This will simply be the identity function
                self.concept_prob_generators.append(
                    lambda x: x
                )
            else:
                self.concept_context_generators.append(
                    torch.nn.Sequential(*([
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            context_gen_out_size,
                        ),
                    ] + act_to_use))
                )
                if self.shared_prob_gen and (
                    len(self.concept_prob_generators) == 0
                ):
                    # Then we will use one and only one probability generator which
                    # will be shared among all concepts. This will force concept
                    # embedding vectors to be pushed into the same latent space
                    self.concept_prob_generators.append(torch.nn.Linear(
                        2 * emb_size,
                        1,
                    ))
                elif not self.shared_prob_gen:
                    self.concept_prob_generators.append(torch.nn.Linear(
                        2 * emb_size,
                        1,
                    ))
        if getattr(self, '_construct_c2y_model', True):
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
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.n_tasks = n_tasks
        self.emb_size = emb_size
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
        **kwargs
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
        # Then time to mix!
        bottleneck = self._construct_c2y_input(
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            probs=output,
            **kwargs,
        )
        return output, intervention_idxs, bottleneck

    def _predict_labels(self, bottleneck, **task_loss_kwargs):
        return self.c2y_model(torch.flatten(bottleneck, start_dim=1, end_dim=-1))

    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        **task_loss_kwargs,
    ):
        if self.prob_training_thresholding > 0 and self.training:
            # Threshold the values in probs with probability
            # self.prob_training_thresholding
            random_tensor = torch.rand(probs.shape).to(probs.device)
            threshold_mask = (random_tensor < self.prob_training_thresholding).float()
            probs = probs * (1 - threshold_mask) + (
                threshold_mask * (probs > 0.5).float()
            )
        bottleneck = (
            pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                neg_embeddings * (
                    1 - torch.unsqueeze(probs, dim=-1)
                )
            )
        )
        bottleneck = bottleneck.view(
            (-1, self.n_concepts * self.emb_size)
        )
        return bottleneck

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

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        pos_embeddings = contexts[:, :, :self.emb_size]
        neg_embeddings = contexts[:, :, self.emb_size:]
        return c_sem, pos_embeddings, neg_embeddings, {}

    def _new_tail_results(
        self,
        x=None,
        c=None,
        y=None,
        c_sem=None,
        bottleneck=None,
        y_pred=None,
    ):
        return []

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
        self._pos_embs = pos_embs
        self._neg_embs = neg_embs


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
                device=x.device,
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
        self._intervention_idxs = intervention_idxs

        y_pred = self._predict_labels(bottleneck=bottleneck)
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

        tail_results += self._new_tail_results(
            x=x,
            c=c,
            y=y,
            c_sem=c_sem,
            bottleneck=bottleneck,
            y_pred=y_pred,
        )
        return tuple([c_sem, bottleneck, y_pred] + tail_results)


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
        if self.l2_penalty > 0.0:
            # Then maximize the l2 distance between positive and negative
            # embeddings for each concept
            l2_loss = 0.0
            for i in range(self.n_concepts):
                l2_loss += torch.mean(
                    torch.square(
                        torch.nn.functional.pairwise_distance(
                            self._pos_embs[:, i, :],
                            self._neg_embs[:, i, :],
                            p=2,
                        )
                    )
                )
            l2_loss = l2_loss / self.n_concepts
            loss += self.l2_penalty * l2_loss

        if self.sim_penalty > 0.0:
            # Then we compute the similarity penalty
            sim_loss = 0.0
            for i in range(self.n_concepts):
                sim_loss += torch.mean(
                    torch.abs(
                        torch.nn.functional.cosine_similarity(
                            self._pos_embs[:, i, :],
                            self._neg_embs[:, i, :],
                            dim=-1,
                        )
                    )
                )
            sim_loss = sim_loss / self.n_concepts
            loss += self.sim_penalty * sim_loss

        if self.emb_pred_loss > 0.0:
            # Then we compute the embedding prediction loss
            emb_pred_loss = 0.0
            for i in range(self.n_concepts):
                c_true = c[:, i].unsqueeze(-1)
                pos_pred = self.emb_pred_layer(self._pos_embs[:, i, :])
                neg_pred = self.emb_pred_layer(self._neg_embs[:, i, :])
                emb_pred_loss += torch.mean(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        pos_pred,
                        c_true,
                    )
                )
                emb_pred_loss += torch.mean(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        neg_pred,
                        c_true,
                    )
                )
            emb_pred_loss = emb_pred_loss / (2 * self.n_concepts)
            loss += self.emb_pred_loss * emb_pred_loss
        if self.prior_loss_term > 0.0:
            # Then we add the task classification loss when the bottleneck
            # is constructed using only the ground-truth concept embeddings
            bottleneck_gt = self._construct_c2y_input(
                pos_embeddings=self._pos_embs,
                neg_embeddings=self._neg_embs,
                probs=c,
            )
            y_pred_gt = self._predict_labels(bottleneck=bottleneck_gt)
            loss += self.prior_loss_term * self.loss_task(
                y_pred_gt,
                y,
            )

        return loss


################################################################################
## Fixed Embedding Version
################################################################################


class FixedEmbConceptEmbeddingModel(ConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
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

        context_gen_out_size=None,

        top_k_accuracy=None,

        # New parameters
        fixed_embeddings=True,
        initial_concept_embeddings=None,
        fixed_embeddings_always=True,
    ):
        """
        Same as a CEM but it has a set of learnable global embeddings
        to use for each concept. Useful if you want the interventions
        to be done using a global set of embeddings.
        """

        super(FixedEmbConceptEmbeddingModel, self).__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            shared_prob_gen=shared_prob_gen,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=None, # KEY!!!!
            inactive_intervention_values=None, # KEY!!!!
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            use_concept_groups=use_concept_groups,
            context_gen_out_size=context_gen_out_size,
            top_k_accuracy=top_k_accuracy,
        )

        # Let's generate the global embeddings we will use
        if (
            (initial_concept_embeddings is None) and
            (active_intervention_values is not None) and
            (inactive_intervention_values is not None)
        ):
            active_intervention_values = torch.tensor(
                active_intervention_values
            )
            inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )

            initial_concept_embeddings = torch.concat(
                [
                    active_intervention_values.unsqueeze(1),
                    inactive_intervention_values.unsqueeze(1),
                ],
                dim=1,
            )
        self.fixed_embeddings_always = fixed_embeddings_always
        self._set_embeddings = True
        if (initial_concept_embeddings is False) or (
            initial_concept_embeddings is None
        ):
            initial_concept_embeddings = torch.normal(
                torch.zeros(self.n_concepts, 2, emb_size),
                torch.ones(self.n_concepts, 2, emb_size),
            )
        else:
            if isinstance(initial_concept_embeddings, np.ndarray):
                initial_concept_embeddings = torch.FloatTensor(
                    initial_concept_embeddings
                )
        self.concept_embeddings = torch.nn.Parameter(
            initial_concept_embeddings,
            requires_grad=(not fixed_embeddings),
        )

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if not self._set_embeddings:
            # Then run the standard CEM pathway
            return ConceptEmbeddingModel._generate_concept_embeddings(
                self=self,
                x=x,
                latent=latent,
                training=training,
            )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent
        if self.fixed_embeddings_always:
            pos_embeddings = self.concept_embeddings[:, 0, :].unsqueeze(0).expand(
                x.shape[0],
                -1,
                -1,
            )
            neg_embeddings = self.concept_embeddings[:, 1, :].unsqueeze(0).expand(
                x.shape[0],
                -1,
                -1,
            )
        else:
            # Else we only use fixed embeddings for interventions
            pos_embeddings = contexts[:, :, :self.emb_size]
            neg_embeddings = contexts[:, :, self.emb_size:]
        return c_sem, pos_embeddings, neg_embeddings, {}


    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        **kwargs
    ):
        if self.fixed_embeddings_always:
            # Then simply use the CEM pathway
            return ConceptEmbeddingModel._after_interventions(
                self=self,
                prob=prob,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                intervention_idxs=intervention_idxs,
                c_true=c_true,
                train=train,
                competencies=competencies,
                **kwargs
            )
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
        # Use the fixed embeddings for the intervened concepts
        global_pos_embs = self.concept_embeddings[:, 0, :].unsqueeze(0).expand(
            intervention_idxs.shape[0],
            -1,
            -1,
        )
        global_neg_embs = self.concept_embeddings[:, 1, :].unsqueeze(0).expand(
            intervention_idxs.shape[0],
            -1,
            -1,
        )
        pos_embeddings = pos_embeddings * (1 - intervention_idxs.unsqueeze(-1)) + (
            intervention_idxs.unsqueeze(-1) * global_pos_embs
        )
        neg_embeddings = neg_embeddings * (1 - intervention_idxs.unsqueeze(-1)) + (
            intervention_idxs.unsqueeze(-1) * global_neg_embs
        )
        # Then time to mix!
        bottleneck = self._construct_c2y_input(
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            probs=output,
            **kwargs,
        )
        return output, intervention_idxs, bottleneck


class LeakyReprCEM(ConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        training_intervention_prob=0,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
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

        # New changes
        sim_penalty=0.0,
        l2_penalty=0.0,
        emb_pred_loss=0.0,
        prob_training_thresholding=0,
        prior_loss_term=0.0,
        cbm_mode=False,
        ##################### end ##########################

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        context_gen_out_size=None,

        top_k_accuracy=None,

        # Parameters for representation definition
        encode_concepts=False,
        encode_tasks=True,
        prob_correct=1.0,
        leak_unprovided_concepts=False,
        leak_provided_concepts=True,
        n_leak_concepts=None,
        prob_encoding_mode="scale",
        prob_flips=0.0,
        neg_base=0,
    ):
        """
        TODO
        """
        if prob_encoding_mode != "fixed":
            assert encode_concepts or encode_tasks, (
                "At least one of encode_concepts or encode_tasks must be True."
            )

        self.encode_concepts = encode_concepts
        self.leak_unprovided_concepts = leak_unprovided_concepts
        self.leak_provided_concepts = leak_provided_concepts
        self.encode_tasks = encode_tasks
        self.prob_correct = prob_correct
        self.n_leak_concepts = n_leak_concepts
        self.prob_encoding_mode = prob_encoding_mode
        self.prob_flips = prob_flips
        self.neg_base = neg_base
        if leak_unprovided_concepts:
            assert n_leak_concepts

        if prob_encoding_mode == "fixed":
            emb_size = n_tasks
        else:
            emb_size = 0
            if encode_concepts:
                if leak_provided_concepts:
                    emb_size += n_concepts
                if leak_unprovided_concepts:
                    emb_size += n_leak_concepts

            if encode_tasks:
                emb_size += n_tasks

            if self.prob_encoding_mode == "dimension":
                emb_size += 1
        super().__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            shared_prob_gen=shared_prob_gen,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            sim_penalty=sim_penalty,
            l2_penalty=l2_penalty,
            emb_pred_loss=emb_pred_loss,
            prob_training_thresholding=prob_training_thresholding,
            prior_loss_term=prior_loss_term,
            cbm_mode=cbm_mode,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            use_concept_groups=use_concept_groups,
            context_gen_out_size=context_gen_out_size,
            top_k_accuracy=top_k_accuracy,
        )

        if self.prob_correct == "predicted":
            self.concept_probes = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    list(
                        self.pre_concept_model.modules()
                    )[-1].out_features,
                    n_concepts,
                )
            )
        if "projection_large" == self.prob_encoding_mode:
            # Then we will have a fixed set of embeddings per concept
            self.M_pos = torch.nn.Parameter(
                torch.randn(n_concepts, emb_size, emb_size),
                requires_grad=False,
            )
            self.M_neg = torch.nn.Parameter(
                torch.randn(n_concepts, emb_size, emb_size),
                requires_grad=False,
            )
        elif "projection" in self.prob_encoding_mode:
            M_pos = torch.randn(emb_size, emb_size)
            self.M_pos = torch.nn.Parameter(
                M_pos + 1e-4 * torch.eye(emb_size),
                requires_grad=(self.prob_encoding_mode == "learned_projection"),
            )
            M_neg = torch.randn(emb_size, emb_size)
            self.M_neg = torch.nn.Parameter(
                M_neg + 1e-4 * torch.eye(emb_size),
                requires_grad=(self.prob_encoding_mode == "learned_projection"),
            )
        elif self.prob_encoding_mode == "fixed":
            # Then we will have a fixed set of embeddings per concept
            self.fixed_pos_embs = torch.nn.Parameter(
                torch.randn(n_concepts, emb_size),
                requires_grad=False,
            )
            self.fixed_neg_embs = torch.nn.Parameter(
                torch.randn(n_concepts, emb_size),
                requires_grad=False,
            )


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

        if isinstance(c, (tuple, list)):
            # Then we are working with a set of provided concepts and unprovided
            # concepts
            c, unprovided_concepts = c
        else:
            unprovided_concepts = None

        if c is None:
            raise ValueError(
                "LeakyReprCEM requires concept labels to be provided "
                "during forward passes."
            )

        if self.leak_unprovided_concepts:
            assert unprovided_concepts is not None, (
                "leak_unprovided_concepts is True, but no unprovided "
                "concepts were given."
            )
            leak_c = unprovided_concepts
        else:
            leak_c = c.clone()
        if self.prob_correct == "predicted":
            if latent is None:
                latent = self.pre_concept_model(x)
            c_sem = self.sig(self.concept_probes(latent))
        elif isinstance(self.prob_correct, (int, float)):
            # Produce the concept probabilities given the ground truth
            if self.prob_correct < 1.0:
                flip_mask = (
                    torch.rand(c.shape).to(x.device) > self.prob_correct
                ).float()
                c_sem = c * (1 - flip_mask) + (1 - c) * flip_mask
            else:
                c_sem = c.clone()

        # Flip the concepts we may use to construct the embedding if we
        # do this with some non-zero chance
        if self.prob_correct == "predicted":
            c_flipped = c_sem.clone().detach()
        elif self.prob_flips > 0:
            flip_mask = (
                torch.rand(c.shape).to(c.device) <= self.prob_flips
            ).float()
            c_flipped = c * (1 - flip_mask) + (1 - c) * flip_mask
        else:
            c_flipped = c.clone()

        if self.prob_flips > 0:
            flip_mask = (
                torch.rand(leak_c.shape).to(leak_c.device) <= self.prob_flips
            ).float()
            leak_c = leak_c * (1 - flip_mask) + (1 - leak_c) * flip_mask

        # Construct the base embedding we will shift to generate our positive
        # and negative embeddings
        base_emb = torch.zeros((c.shape[0], self.emb_size)).float().to(
            c.device
        )

        offset = 0
        if self.prob_encoding_mode == "dimension":
            end_offset = self.emb_size - 1
        else:
            end_offset = self.emb_size

        if self.encode_concepts:
            # Then the first dimensions of the embedding will correspond to the
            # concept labels
            if self.leak_unprovided_concepts:
                if not self.leak_provided_concepts:
                    offset = self.n_leak_concepts
                    base_emb[:, :self.n_leak_concepts] = leak_c
                else:
                    # Else we leak all concepts!
                    offset = self.n_concepts + self.n_leak_concepts
                    base_emb[:, :offset] = torch.concat(
                        [c_flipped, leak_c],
                        dim=-1,
                    )
            else:
                offset = self.n_concepts
                base_emb[:, :self.n_concepts] = c_flipped
        # print("c[0, :] =", c[0, :])
        # print("c_sem[0, :] =", c_sem[0, :])
        # print("base_emb[0, :] =", base_emb[0, :])


        if self.encode_tasks:
            if y is None:
                raise ValueError(
                    "LeakyReprCEM requires task labels to be provided "
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
                flip_mask = (
                    torch.rand(y.shape).to(y.device) <= self.prob_flips
                ).long()
                leak_y = y * (1 - flip_mask) + random_labels * flip_mask
            else:
                leak_y = y
            # Then we will encode the tasks as one hot encodings in the embedding
            one_hot_y = torch.nn.functional.one_hot(
                leak_y,
                num_classes=self.n_tasks,
            )
            base_emb[:, offset:end_offset] = one_hot_y

        if self.prob_encoding_mode == "fixed":
            pos_embs = self.fixed_pos_embs.unsqueeze(0).expand(
                c.shape[0],
                -1,
                -1,
            ).clone()
            neg_embs = self.fixed_neg_embs.unsqueeze(0).expand(
                c.shape[0],
                -1,
                -1,
            ).clone()
        else:
            base_emb = base_emb.unsqueeze(1).expand(-1, self.n_concepts, -1).clone()
            if self.encode_concepts and self.leak_provided_concepts:
                # Then we make sure to drop the probability of the concept
                # correspoinding to the representation itself (as it will be
                # provided elsewhere and it could be mistaken)
                for concept_idx in range(self.n_concepts):
                    base_emb[:, concept_idx, concept_idx] = -1


            if self.prob_encoding_mode == "dimension":
                base_emb[:, :, -1] = c_sem
                pos_embs = base_emb
                neg_embs = 1 - pos_embs
                # cp + (1-c)(-p) = cp - p + cp = p(2c-1)
            if self.prob_encoding_mode == "dimension_new":
                base_emb[:, :, -1] = 1
                pos_embs = base_emb
                neg_embs =  -(base_emb + self.neg_base)
                neg_embs[:, :, -1] = 0  # Asymmetry
                # cp + (1-c)(-p) = cp - p + cp = p(2c-1)
            if "projection_large" in self.prob_encoding_mode:
                # We will then multiply self.M_pos, a (n_concepts, emb_size, emb_size) matrx
                # with the base embedding, a (batch_size, n_concepts, emb_size) matrix, to get
                # the positive embeddings with shape (batch_size, n_concepts, emb_size)
                M_pos = self.M_pos.to(base_emb.device)
                pos_embs = torch.einsum(
                    "bce,ceh->bch",
                    base_emb,
                    M_pos,
                ).to(base_emb.device)
                M_neg = self.M_neg.to(base_emb.device)
                neg_embs = torch.einsum(
                    "bce,ceh->bch",
                    base_emb,
                    M_neg,
                ).to(base_emb.device)
            elif "projection" in self.prob_encoding_mode:
                # Then project the base embedding into the pos and negative spaces
                # using the random invertible matrices
                M_pos = self.M_pos.to(base_emb.device)
                pos_embs = base_emb @ M_pos.T
                M_neg = self.M_neg.to(base_emb.device)
                neg_embs = base_emb @ M_neg.T
            else:
                pos_embs = base_emb
                neg_embs = -(base_emb + self.neg_base)
                # cp + (1 - c)(-p - B) = cp - p + cp - B + Bc= 2cp - p - B(c-1) = p(2c - 1) - B(c-1)

        if self.prob_correct == "predicted_cem":
            if latent is None:
                latent = self.pre_concept_model(x)
            c_sem = []
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(latent)
                # context = torch.cat(
                #     [pos_embs[:, i, :], neg_embs[:, i, :]],
                #     dim=-1,
                # )
                prob = prob_gen(context)
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)

        self._pos_embs = pos_embs
        self._neg_embs = neg_embs

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
                device=x.device,
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
        )
        self._intervention_idxs = intervention_idxs

        y_pred = self._predict_labels(bottleneck=bottleneck)
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
            tail_results.append(latent)
        if output_embeddings and (not pos_embs is None) and (
            not neg_embs is None
        ):
            tail_results.append(pos_embs)
            tail_results.append(neg_embs)

        tail_results += self._new_tail_results(
            x=x,
            c=c,
            y=y,
            c_sem=c_sem,
            bottleneck=bottleneck,
            y_pred=y_pred,
        )
        return tuple([c_sem, bottleneck, y_pred] + tail_results)

    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        **task_loss_kwargs,
    ):
        if self.prob_training_thresholding > 0 and self.training:
            # Threshold the values in probs with probability
            # self.prob_training_thresholding
            random_tensor = torch.rand(probs.shape).to(probs.device)
            threshold_mask = (random_tensor < self.prob_training_thresholding).float()
            probs = probs * (1 - threshold_mask) + (
                threshold_mask * (probs > 0.5).float()
            )
        if self.prob_encoding_mode == "dimension":
            bottleneck = pos_embeddings.clone()
            bottleneck[:, :, -1] = probs
        else:
            bottleneck = (
                pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                    neg_embeddings * (
                        1 - torch.unsqueeze(probs, dim=-1)
                    )
                )
            )
        bottleneck = bottleneck.reshape(
            (-1, self.n_concepts * self.emb_size)
        )
        return bottleneck