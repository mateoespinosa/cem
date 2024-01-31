import numpy as np
import torch
import logging
from cem.interventions.coop import CooP

class FlowInterventionPolicy(CooP):
    def __init__(
        self,
        concept_group_map,
        cbm,
        n_tasks,
        num_groups_intervened=0,
        importance_weight=1,
        acquisition_weight=1,
        acquisition_costs=None,
        group_based=True,
        eps=1e-8,
        **kwargs
    ):
        kwargs.pop("concept_entropy_weight", 0)
        kwargs.pop("include_prior", False)
        CooP.__init__(
            self,
            num_groups_intervened=num_groups_intervened,
            concept_group_map=concept_group_map,
            cbm=cbm,
            concept_entropy_weight=0,
            importance_weight=importance_weight,
            acquisition_costs=acquisition_costs,
            acquisition_weight=acquisition_weight,
            group_based=group_based,
            eps=eps,
            include_prior=True,
            n_tasks=n_tasks,
            **kwargs,
        )

    def _importance_score(
        self,
        x,
        c,
        pred_c,
        prev_interventions,
        prev_entropy,
        latent=None,
        competencies=None,
        prior_distribution=None,
        intervene_with_groups=False,
    ):


        if intervene_with_groups:
            for group_idx, (_, group_concepts) in enumerate(
                self.concept_group_map.items()
            ):
                if group_idx == concept_group_idx:
                    break
        else:
            group_concepts = [concept_group_idx]
        # And then estimating how much this would change if we intervene on the
        # concept of interest
        expected_new_entropy = torch.zeros((x.shape[0],)).to(pred_c.device)
        for concept_val_idx in range(len(group_concepts)):
            for val_to_set in [0, 1] if len(group_concepts) == 1 else [1]:
                old_c_vals = c[:, group_concepts]
                # Make a fake intervention in the current concept
                new_vals = torch.zeros((x.shape[0], len(group_concepts))).to(
                    c.device
                )
                new_vals[:, concept_val_idx] = val_to_set
                c[:, group_concepts] = new_vals
                prev_mask = prev_interventions[:, group_concepts] == 1
                prev_interventions[:, group_concepts] = 1
                # See how the predictions change
                _, _, change_y_preds, _, _ = self.cbm(
                    x,
                    intervention_idxs=prev_interventions,
                    c=c,
                    latent=latent,
                )
                # Restore prev-interventions
                c[:, group_concepts] = old_c_vals
                prev_interventions[:, group_concepts] = prev_mask.type(
                    prev_interventions.type()
                )
                # And compute their weighted output
                prob = pred_c[:, group_concepts[concept_val_idx]]
                if val_to_set == 0:
                    prob = 1 - prob

                # And scale the probability by our intervention prior
                if self.n_tasks > 1:
                    new_class_probs = torch.nn.functional.softmax(
                        change_y_preds,
                        dim=-1,
                    )

                else:
                    y_probs = torch.sigmoid(change_y_preds)
                    new_class_probs = torch.cat(
                        [1 - y_probs, y_probs],
                        dim=-1,
                    )
                expected_new_entropy += \
                    prior_distribution[:, concept_group_idx] * (
                        prob * torch.sum(
                            -torch.log(new_class_probs + self.eps) * \
                                new_class_probs,
                            axis=-1,
                        )
                    )
        return -(
            expected_new_entropy +
            (1 - prior_distribution[:, concept_group_idx]) * prev_entropy
        )

    def _coop_step(
        self,
        x,
        c,
        pred_c,
        prev_interventions,
        latent,
        prev_entropy,
        y=None,
        y_preds=None,
        competencies=None,
        prior_distribution=None,
        intervene_with_groups=True,
    ):
        if intervene_with_groups and self.cbm.use_concept_groups:
            n_groups = len(self.concept_group_map)
            prev_used_groups = torch.zeros((c.shape[0], n_groups)).to(c.device)
            for group_idx, (_, group_concepts) in enumerate(
                self.concept_group_map.items()
            ):
                prev_used_groups[:, group_idx] = torch.any(
                    prev_interventions[:, group_concepts] == 1,
                    dim=-1,
                )
        else:
            n_groups = c.shape[-1]
            prev_used_groups = prev_interventions
            if self.cbm.use_concept_groups and (not prior_distribution is None):
                new_prior = torch.zeros((c.shape[0], n_groups)).to(c.device)
                for group_idx, (_, group_concepts) in enumerate(
                    self.concept_group_map.items()
                ):
                    new_prior[:, group_concepts] = (
                        prior_distribution[:, group_idx]
                    )
                prior_distribution = new_prior
        if prior_distribution is None:
            denom = torch.sum(prev_used_groups, dim=-1, keepdim=True)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            prior_distribution = torch.ones(prev_used_groups.shape).to(
                prev_used_groups.device
            ) / denom

        sample_importances = torch.zeros((c.shape[0], n_groups)).to(
            pred_c.device
        )
        for concept_group_idx in range(n_groups):
            sample_importances[:, concept_group_idx] = self._importance_score(
                x=x,
                concept_group_idx=concept_group_idx,
                c=c,
                pred_c=pred_c,
                prev_interventions=prev_interventions,
                prev_entropy=prev_entropy,
                latent=latent,
                competencies=competencies,
                prior_distribution=prior_distribution,
                intervene_with_groups=intervene_with_groups,
            )
        if isinstance(prev_used_groups, np.ndarray):
            prev_used_groups = torch.from_numpy(prev_used_groups).to(c.device)
        sample_importances = torch.where(
            prev_used_groups == 1,
            torch.ones_like(sample_importances) * (-np.Inf),
            sample_importances,
        )
        if not self.cbm.use_concept_groups:
            new_sample_importances = torch.zeros(
                (c.shape[0], len(self.concept_group_map))
            ).to(c.device)
            for concept_group_idx, (_, group_concepts) in enumerate(
                self.concept_group_map.items()
            ):
                new_sample_importances[:, concept_group_idx] = torch.mean(
                    sample_importances[:, group_concepts],
                    dim=-1,
                )
            sample_importances = new_sample_importances
        # Updates prev_interventions rather than copying it to speed things up
        if self.group_based:
            group_names = []
            for _, (group_name, _) in enumerate(
                self.concept_group_map.items()
            ):
                group_names.append(group_name)
            next_groups = torch.argmax(sample_importances, axis=-1)
            for sample_idx, best_group_idx in enumerate(next_groups):
                # Get the concepts corresponding to the group we will be
                # intervening on
                next_concepts = self.concept_group_map[
                    group_names[best_group_idx]
                ]
                prev_interventions[sample_idx, next_concepts] = 1
        else:
            raise ValueError("Not implemented")
            next_concepts = torch.argmax(sample_scores, axis=-1)
            prev_interventions[:, next_concepts.detach().cpu().numpy()] = 1

        return prev_interventions, latent, y_preds

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
            mask = torch.zeros((x.shape[0], c.shape[-1])).to(c.device)
        else:
            mask = prev_interventions[:]

        if not self.include_prior:
            prior_distribution = None


        if self.group_based and self.cbm.use_concept_groups:
            n_groups = len(self.concept_group_map)
            prev_used_groups = torch.zeros((c.shape[0], n_groups)).to(c.device)
            for group_idx, (_, group_concepts) in enumerate(
                self.concept_group_map.items()
            ):
                prev_used_groups[:, group_idx] = torch.any(
                    mask[:, group_concepts] == 1,
                    dim=-1,
                )
        else:
            n_groups = c.shape[-1]
            prev_used_groups = mask
            if self.cbm.use_concept_groups and (not prior_distribution is None):
                new_prior = torch.zeros((c.shape[0], n_groups)).to(c.device)
                for group_idx, (_, group_concepts) in enumerate(
                    self.concept_group_map.items()
                ):
                    new_prior[:, group_concepts] = (
                        prior_distribution[:, group_idx]
                    )
                prior_distribution = new_prior
        if prior_distribution is None:
            denom = torch.sum(prev_used_groups, dim=-1, keepdim=True)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            prior_distribution = torch.ones(prev_used_groups.shape).to(
                prev_used_groups.device
            ) / denom

        if competencies is None:
            competencies = torch.ones(c.shape).to(x.device)
        if self.num_groups_intervened == len(self.concept_group_map):
            return np.ones(c.shape, dtype=np.int64), c
        if self.num_groups_intervened == 0:
            return mask, c
        _, _, y_preds, _, latent = self.cbm(
            x,
            intervention_idxs=mask,
            c=c,
        )
        if self.n_tasks > 1:
            pred_class_probs = torch.nn.functional.softmax(y_preds, dim=-1)
        else:
            y_probs = torch.sigmoid(y_preds)
            pred_class_probs = torch.cat(
                [1 - y_probs, y_probs],
                dim=-1,
            )
        prev_entropy = torch.sum(
            -torch.log(pred_class_probs + self.eps) * pred_class_probs,
            dim=-1,
        )

        for i in range(self.num_groups_intervened):
            logging.debug(
                f"Intervening with {i + 1}/{self.num_groups_intervened} "
                f"concepts in CooP"
            )
            mask, latent, y_preds = self._coop_step(
                x=x,
                c=c,
                pred_c=pred_c,
                prev_interventions=mask,
                y=y,
                latent=latent,
                y_preds=y_preds,
                competencies=competencies,
                prior_distribution=prior_distribution,
                prev_entropy=prev_entropy,
                intervene_with_groups=self.group_based,
            )
        return mask, c