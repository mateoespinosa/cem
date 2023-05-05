import numpy as np
import torch

class IndependentRandomMaskIntPolicy(object):

    def __init__(
        self,
        cbm,
        concept_group_map,
        num_groups_intervened=0,
        group_based=True,
        include_prior=True,
        horizon=1,
        **kwargs,
    ):
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.group_based = group_based
        self.include_prior = include_prior
        self.horizon = horizon
        self.cbm_use_concept_groups = cbm.use_concept_groups

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
        # We have to split it into a list contraction due to the
        # fact that we can't afford to run a np.random.choice
        # that does not allow replacement between samples...
        if self.group_based:
            concept_group_map = self.concept_group_map
        else:
            concept_group_map = dict([
                (i, [i]) for i in range(c.shape[-1])
            ])
        if not self.include_prior:
            prior_distribution = None
        if prev_interventions is not None:
            mask = prev_interventions.detach().cpu().numpy()
        else:
            mask = np.zeros((x.shape[0], c.shape[-1]), dtype=np.int64)
        group_mask = np.zeros((x.shape[0], len(concept_group_map)))
        for group_idx, (_, group_concepts) in enumerate(concept_group_map.items()):
            group_mask[:, group_idx] = np.all(mask[:, group_concepts] > 0, axis=-1).astype(np.int64)
        prev_intervened_groups = np.sum(group_mask, axis=-1)
        if prior_distribution is None:
            prior_distribution = np.array([
                [
                    1/(
                        (len(self.concept_group_map) - prev_intervened_groups[idx])
                        if prev_intervened_groups[idx] else len(self.concept_group_map)
                    )
                    if group_mask[idx, group_idx] == 0 else 0
                    for group_idx in range(group_mask.shape[-1])
                ]
                for idx in range(x.shape[0])
            ])
        else:
            prior_distribution = prior_distribution.detach().cpu().numpy()
            if not self.cbm_use_concept_groups:
                prior_distribution = np.array([
                    [
                        np.sum(prior_distribution[idx, concepts])
                        for _, concepts in self.concept_group_map.items()
                    ]
                    for idx in range(x.shape[0])
                ])
        selected_groups_for_trial = np.array([
            np.random.choice(
                list(self.concept_group_map.keys()),
                size=self.num_groups_intervened,
                replace=False,
                p=prior_distribution[idx, :],
            ) for idx in range(x.shape[0])
        ])
        for sample_idx in range(selected_groups_for_trial.shape[0]):
            for selected_group in selected_groups_for_trial[sample_idx, :]:
                mask[sample_idx, self.concept_group_map[selected_group]] = 1
        return mask, c
