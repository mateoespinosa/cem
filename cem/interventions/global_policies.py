import numpy as np
import torch
import io
from contextlib import redirect_stdout
import pytorch_lightning as pl
from cem.interventions.intervention_policy import InterventionPolicy

class ConstantMaskPolicy(InterventionPolicy):
    def __init__(
        self,
        mask,
        concept_group_map=None,
        num_groups_intervened=0,
        group_based=True,
        include_prior=True,
        **kwargs,
    ):
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.group_based = group_based
        self.include_prior = include_prior
        self.mask = mask

    def __call__(
        self,
        x,
        pred_c,
        c,
        y=None,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
    ):
        return torch.tile(
            torch.unsqueeze(
                torch.FloatTensor(self.mask).to(c.device),
                0,
            ),
            (c.shape[0], 1),
        ), c

class GlobalValidationPolicy(InterventionPolicy):
    # Intervenes first on concepts with the highest uncertainty (measured by
    # their predicted distribution's entropy)
    # Adapted from the ideas in https://openreview.net/pdf?id=PUspzfGsgY
    def __init__(
        self,
        concept_group_map,
        val_c_aucs,
        num_groups_intervened=0,
        group_based=True,
        eps=1e-8,
        include_prior=True,
        **kwargs,
    ):
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.eps = eps
        self.group_based = group_based
        self.include_prior = include_prior
        # We have to split it into a list contraction due to the
        # fact that we can't afford to run a np.random.choice
        # that does not allow replacement between samples...
        self.scores = 1 / np.abs(val_c_aucs - 0.5 + self.eps)

    def __call__(
        self,
        x,
        pred_c,
        c,
        y=None,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
    ):
        if prev_interventions is None:
            mask = np.zeros((x.shape[0], c.shape[-1]), dtype=np.int64)
        else:
            prev_interventions = prev_interventions.detach().cpu().numpy()
            mask = prev_interventions
        if not self.include_prior:
            prior_distribution = None
        elif prior_distribution is not None:
            prior_distribution = prior_distribution.detach().cpu().numpy()

        scores = np.tile(
            np.expand_dims(self.scores, 0),
            (c.shape[0], 1),
        )

        if prior_distribution is not None:
            # Then rescale the scores based on the prior
            scores *= prior_distribution
        if prev_interventions is not None:
            # Then zero out the scores of the concepts that have been previously
            # intervened
            scores = np.where(
                prev_interventions == 1,
                -float("inf"),
                scores,
            )

        best_concepts = np.argsort(-scores, axis=-1)
        for sample_idx in range(c.shape[0]):
            if self.group_based:
                # We will assign each group a score based on the max score of
                # its corresponding concepts
                group_scores = np.zeros(len(self.concept_group_map))
                group_names = []
                for i, key in enumerate(self.concept_group_map):
                    group_scores[i] = np.max(
                        scores[sample_idx, self.concept_group_map[key]],
                        axis=-1,
                    )
                    group_names.append(key)
                # Sort them out
                best_group_scores = np.argsort(-group_scores, axis=-1)
                for selected_group in (
                    best_group_scores[: self.num_groups_intervened]
                ):
                    mask[
                        sample_idx,
                        self.concept_group_map[group_names[selected_group]]
                    ] = 1

            else:
                # Else, previous interventions do not affect future ones
                mask[
                    sample_idx,
                    best_concepts[sample_idx, : self.num_groups_intervened]
                ] = 1
        return mask, c

class GlobalValidationImprovementPolicy(GlobalValidationPolicy):
    # Intervenes first on concepts with the highest uncertainty (measured by
    # their predicted distribution's entropy)
    # Adapted from the ideas in https://openreview.net/pdf?id=PUspzfGsgY
    def __init__(
        self,
        cbm,
        concept_group_map,
        n_concepts,
        val_ds,
        num_groups_intervened=0,
        group_based=True,
        eps=1e-8,
        include_prior=True,
        **kwargs,
    ):
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.eps = eps
        self.group_based = group_based
        self.include_prior = include_prior

        prev_policy = cbm.intervention_policy
        cbm.intervention_policy = None
        self.scores = np.zeros((n_concepts,))
        trainer = pl.Trainer(
            gpus=int(torch.cuda.is_available()),
            logger=False,
        )
        f = io.StringIO()
        for concept_idx in range(n_concepts):
            mask = np.zeros((n_concepts,))
            mask[concept_idx] = 1
            cbm.intervention_policy = ConstantMaskPolicy(
                mask=mask,
            )
            with redirect_stdout(f):
                [val_results] = trainer.test(cbm, val_ds)
            self.scores[concept_idx] = val_results["test_y_accuracy"]
        cbm.intervention_policy = prev_policy