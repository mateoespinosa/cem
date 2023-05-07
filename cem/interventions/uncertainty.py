import numpy as np
import torch
from cem.interventions.intervention_policy import InterventionPolicy

class UncertaintyMaximizerPolicy(InterventionPolicy):
    # Intervenes first on concepts with the highest uncertainty (measured by their
    # predicted distribution's entropy)
    # Adapted from the ideas in https://openreview.net/pdf?id=PUspzfGsgY
    def __init__(
        self,
        concept_group_map,
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
            mask = prev_interventions.detach().cpu().numpy()
        if not self.include_prior:
            prior_distribution = None
        # We have to split it into a list contraction due to the
        # fact that we can't afford to run a np.random.choice
        # that does not allow replacement between samples...
        scores = 1 / np.abs(pred_c.cpu().detach().numpy() - 0.5 + self.eps)
        if prev_interventions is not None:
            # Then zero out the scores of the concepts that have been previously intervened
            scores[prev_interventions.type(torch.BoolTensor)] = 0
        if prior_distribution is not None:
            # Then rescale the scores based on the prior
            scores *= prior_distribution.detach().cpu().numpy()
        best_concepts = np.argsort(-scores, axis=-1)
        for sample_idx in range(c.shape[0]):
            if self.group_based:
                # We will assign each group a score based on the max score of its
                # corresponding concepts
                group_scores = np.zeros(len(self.concept_group_map))
                group_names = []
                for i, key in enumerate(self.concept_group_map):
                    group_scores[i] = np.max(scores[sample_idx, self.concept_group_map[key]], axis=-1)
                    group_names.append(key)
                # Sort them out
                best_group_scores = np.argsort(-group_scores, axis=-1)
                for selected_group in best_group_scores[: self.num_groups_intervened]:
                    mask[sample_idx, self.concept_group_map[group_names[selected_group]]] = 1

            else:
                # Else, previous interventions do not affect future ones
                mask[sample_idx, best_concepts[sample_idx, : self.num_groups_intervened]] = 1
        return mask, c


