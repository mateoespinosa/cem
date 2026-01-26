import numpy as np
import pytorch_lightning as pl
import torch

from cem.metrics.accs import compute_accuracy
from cem.models.cbm import ConceptBottleneckModel


################################################################################
## Posthoc CBM
################################################################################


class PCBM(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size,
        pretrained_model,
        concept_vectors=None,
        concept_vector_intercepts=None,

        c2y_model=None,
        output_latent=False,
        residual=False,
        residual_model=None,
        residual_use=True,
        reg_strength=1e-5, # regularization strenght for downstream sparse classifier
        l1_ratio=0.99,
        freeze_pretrained_model=True,
        freeze_concept_embeddings=True,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,
        training_intervention_prob=0.0,

        top_k_accuracy=None,

        # New additions
        prior_loss_term=0.0,
        default_intervention_bound=1,

    ):
        """
        Implementation of a Post-hoc CBM by Yuksekgonul et al. (https://arxiv.org/abs/2205.15480).
        For this implementation, we tried to stick as closely as possible to the
        original implementation of the authors (found here: https://github.com/mertyg/post-hoc-cbm),
        changing things only so that the model fits the interfaces used in our
        experimentation pipeline.
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.top_k_accuracy = top_k_accuracy
        self.concept_loss_weight = 0
        self.extra_dims = 0
        self.sigmoidal_prob = False

        # Let's first set up the embedding extractor from the pretrained model
        # The assumption here is that this model already generates the embedding
        # (i.e., it has been modified so that is output is the layer we are
        # targeting)
        self.embedding_generator = pretrained_model
        self.freeze_pretrained_model = freeze_pretrained_model
        if freeze_pretrained_model:
            # Then let's go ahead and freeze all the parameters in this model
            for param in self.embedding_generator.parameters():
                param.requires_grad = False

        # Time to initialize our embedding projection matrix
        self.freeze_concept_embeddings = freeze_concept_embeddings
        self.emb_size = emb_size
        if concept_vectors is not None:
            assert concept_vectors.shape == (n_concepts, emb_size), (
                f'Expected the concept vector matrix to be of shape '
                f'(n_concepts, emb_size) however we got a matrix of size '
                f'{concept_vectors.shape} even though we were told we would expect '
                f'{self.n_concepts} concepts with shape {emb_size}.'
            )
            self.concept_embeddings = torch.nn.Parameter(
                concept_vectors,
                requires_grad=(not freeze_concept_embeddings),
            )
        else:
            self.concept_embeddings = torch.nn.Parameter(
                torch.rand((self.n_concepts, self.emb_size)),
                requires_grad=(not freeze_concept_embeddings),
            )
        # Precompute the norms as we will use them during inference
        self.concept_norms = torch.norm(
            self.concept_embeddings,
            p=2,
            dim=1,
            keepdim=False,
        )
        # Handle intercepts (if provided)
        if concept_vector_intercepts is None:
            concept_vector_intercepts = torch.zeros(
                (n_concepts,)
            )
        self.concept_intercepts = torch.nn.Parameter(
            concept_vector_intercepts,
            requires_grad=(not freeze_concept_embeddings),
        )


        # Now construct the downstream interpretable model
        if c2y_model is None:
            self._c2y_model_provided = False
            self.c2y_model = torch.nn.Linear(self.n_concepts, self.n_tasks)
        else:
            self._c2y_model_provided = True
            self.c2y_model = c2y_model

        # Scafolding needed for intervention support here
        self.output_latent = output_latent
        self.use_concept_groups = use_concept_groups
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.training_intervention_prob = training_intervention_prob
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = \
                default_intervention_bound * torch.ones(n_concepts)

        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = \
                -default_intervention_bound * torch.ones(n_concepts)



        # Set up the residual model, if any
        self.residual = residual
        self.residual_use = residual_use
        self.residual_model = None
        if self.residual:
            if residual_model is None:
                residual_model = torch.nn.Linear(
                    emb_size,
                    n_tasks,
                )
            self.residual_model = residual_model

        # And the losses. Notice we will still have a concept loss so that
        # we can use the CBM base class but this will be a no-op
        self.loss_concept = lambda *args, **kwargs: 0.0
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.reg_strength = reg_strength
        self.l1_ratio = l1_ratio

        # Finally, stuff for the actual optimization
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.prior_loss_term = prior_loss_term


    def _generate_concept_scores(
        self,
        x,
        latent=None,
        training=False,
    ):
        if latent is None:
            latent = self.embedding_generator(x)
        if isinstance(latent, dict):
            if len(latent) == 1:
                latent_key = list(latent.keys())[0]
            else:
                raise ValueError(
                    'Currently only support dictionaries of length 1 for '
                    'embedding_generator and instead got an output of size '
                    f'{len(latent)}'
                )
            latent = latent[latent_key]
        # Embeddings are (B, m) and concept vectors are (k, m)
        projections = torch.matmul(latent, self.concept_embeddings.T)
        projections = (projections + self.concept_intercepts.unsqueeze(0))/self.concept_norms.unsqueeze(0).to(projections.device)
        return projections, latent


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
        if self._c2y_model_provided:
            # Then we use all weights for the regularization
            l1_norm = 0.0
            l2_norm = 0.0
            for param in self.c2y_model.parameters():
                if param.requires_grad:
                    l1_norm += torch.norm(param, p=1)
                    l2_norm += torch.pow(param, 2)
            l2_norm = torch.square(l2_norm)
        else:
            # Otherwise this is the detault sparse network for which we only
            # need to normalize its weight vector
            l1_norm = torch.norm(self.c2y_model.weight, p=1)
            l2_norm = torch.norm(self.c2y_model.weight, p=2)
        elastic_net = l1_norm * self.l1_ratio + (1 - self.l1_ratio) * l2_norm

        # And normalize by classes and concepts while also considering the
        # regularizer strength
        loss = elastic_net * self.reg_strength / (
            self.n_concepts * self.n_tasks
        )

        if self.prior_loss_term > 0.0:
            new_y_pred = self._forward(
                x=x,
                intervention_idxs=torch.ones_like(c),
                c=c,
                y=y,
                train=self.training,
                competencies=competencies,
                prev_interventions=None,
            )[2]
            loss += self.prior_loss_term * self.loss_task(
                new_y_pred if new_y_pred.shape[-1] > 1 else new_y_pred.reshape(-1),
                y,
            )

        return loss

    def freeze_non_residual_components(self):
        if not self.freeze_pretrained_model:
            # Then let's go ahead and freeze all the parameters in this model
            for param in self.embedding_generator.parameters():
                param.requires_grad = False
        if not self.freeze_concept_embeddings:
            self.concept_embeddings.requires_grad = False
            self.concept_intercepts.requires_grad = False
        if (self.residual_model is not None):
            for param in self.residual_model.parameters():
                param.requires_grad = True

    def unfreeze_non_residual_components(self):
        if not self.freeze_pretrained_model:
            # Then let's go ahead and freeze all the parameters in this model
            for param in self.embedding_generator.parameters():
                param.requires_grad = True
        if not self.freeze_concept_embeddings:
            self.concept_embeddings.requires_grad = True
            self.concept_intercepts.requires_grad = True

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

        c_pred, latent = self._generate_concept_scores(
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
            active_intervention_values = \
                self.active_intervention_values.to(
                    c_pred.device
                )
            pos_embs = torch.tile(
                active_intervention_values,
                (c.shape[0], 1),
            ).to(active_intervention_values.device)
            inactive_intervention_values = \
                self.inactive_intervention_values.to(
                    c_pred.device
                )
            neg_embs = torch.tile(
                inactive_intervention_values,
                (c.shape[0], 1),
            ).to(inactive_intervention_values.device)

            pos_embs = torch.unsqueeze(pos_embs, dim=-1)
            neg_embs = torch.unsqueeze(neg_embs, dim=-1)

            prior_distribution = self._prior_int_distribution(
                prob=c_pred,
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
                pred_c=c_pred,
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

        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )

        y_pred = self.c2y_model(c_pred)
        # Then time to consider the residual
        if self.residual_use and (self.residual_model is not None):
            y_pred += self.residual_model(latent)
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
        return tuple([c_pred, c_pred, y_pred] + tail_results)


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
        active_intervention_values = self.active_intervention_values.to(
            c_pred.device
        )
        batched_active_intervention_values = torch.unsqueeze(
            active_intervention_values,
            0,
        ).expand(c_pred.shape[0], -1)

        inactive_intervention_values = self.inactive_intervention_values.to(
            c_pred.device
        )
        batched_inactive_intervention_values = torch.unsqueeze(
            inactive_intervention_values,
            0,
        ).expand(c_pred.shape[0], -1)

        ground_truth_vals = (
            (
                c_true *
                batched_active_intervention_values
            ) +
            (
                (c_true - 1) *
                -batched_inactive_intervention_values
            )
        )
        c_pred_copy = ground_truth_vals * intervention_idxs + (
            torch.logical_not(intervention_idxs) * c_pred
        )
        return c_pred_copy


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
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        task_loss = self.loss_task(
            y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
            y,
        )
        task_loss_scalar = task_loss.detach()
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
        # compute accuracy
        _, (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_pred=None,
            y_pred=y_logits,
            c_true=None,
            y_true=y,
        )
        result = {
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
        }
        return loss, result
