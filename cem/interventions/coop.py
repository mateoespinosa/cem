import numpy as np
import torch
import logging
from cem.interventions.intervention_policy import InterventionPolicy

##########################
## CooP Policy Definition
##########################

class CooP(InterventionPolicy):
    # CooP Intervention Policy
    def __init__(
        self,
        concept_group_map,
        cbm,
        n_tasks,
        num_groups_intervened=0,
        concept_entropy_weight=1,
        importance_weight=1,
        acquisition_weight=1,
        acquisition_costs=None,
        group_based=True,
        eps=1e-8,
        max_uncertainty=None,
        min_uncertainty=None,
        max_importance=None,
        min_importance=None,
        use_uncertainty_proxy=False,
        include_prior=False,
        **kwargs
    ):
        self.n_tasks = n_tasks
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.concept_entropy_weight = concept_entropy_weight
        self.importance_weight = importance_weight
        self.use_uncertainty_proxy = use_uncertainty_proxy
        if acquisition_costs is None:
            self.acquisition_costs = acquisition_costs
        else:
            self.acquisition_costs = acquisition_weight * acquisition_costs
        self.group_based = group_based
        self._optimal = False
        self.cbm = cbm
        self.eps = eps
        if max_uncertainty is None:

            if use_uncertainty_proxy:
                # Then the maximum uncertainty will be when
                # the predicted probability is 0.5, which is unbounded...
                # so we will set it to be equal to 1 for now and leave it
                # unbounded as well
                max_uncertainty = 1
            else:
                # Then this will be -ln(1/2) as this is the
                # entropy (in nats) of a binary uniform
                # distribution.
                max_uncertainty = -np.log(0.5)
        self.max_uncertainty = max_uncertainty
        if min_uncertainty is None:
            # Then this will be -ln(1) as this is the
            # entropy (in nats) of a delta function
            # distribution.
            min_uncertainty = 0
        self.min_uncertainty = min_uncertainty
        if max_importance is None:
            # Then this will correspond to 1 as that
            # how much the probability of a given class
            # can change (from 0 to 1)
            max_importance = 1
        self.max_importance = max_importance
        if min_importance is None:
            # Then this will correspond to 0 as the
            # probability of a given class can remain
            # the same after a concept was intervened on
            min_importance = 0
        self.min_importance = min_importance
        self.include_prior = include_prior

    def _normalize_importance(self, importance_scores):
        if (
            (self.max_importance is None) or
            (self.min_importance is None)
        ):
            # Then nothing to normalize here!
            return importance_scores
        return (importance_scores - self.min_importance) / (
            self.max_importance - self.min_importance
        )

    def _normalize_uncertainty(self, uncertainty_scores):
        if (
            (self.max_uncertainty is None) or
            (self.min_uncertainty is None)
        ):
            # Then nothing to normalize here!
            return uncertainty_scores
        return (uncertainty_scores - self.min_uncertainty) / (
            self.max_uncertainty - self.min_uncertainty
        )

    def _importance_score(
        self,
        x,
        concept_idx,
        c,
        pred_c,
        prev_interventions,
        latent=None,
        pred_class_prob=None,
        pred_class=None,
        competencies=None,
        prior_distribution=None,
    ):
        hat_c = pred_c[:, concept_idx]
        # Find the class we will be predicting
        if (pred_class is None) or (pred_class_prob is None):
            _, _, y_preds, _, latent = self.cbm(
                x,
                intervention_idxs=prev_interventions,
                c=c,
                latent=latent,
            )
            if self.n_tasks > 1:
                pred_class_prob, pred_class = torch.nn.functional.softmax(
                    y_preds,
                    dim=-1
                ).max(dim=-1)
            else:
                y_probs = torch.sigmoid(y_preds)
                pred_class = torch.squeeze((y_probs >= 0.5), axis=-1).type(
                    y_preds.type()
                )
                pred_class_prob = (
                    y_probs * pred_class + (1 - y_probs) * (1 - pred_class)
                )



        # And then estimating how much this would change if we intervene on the
        # concept of interest
        if self._optimal:
            # Then just look at the value of the probability of the known truth
            # class, as this is what we want to maximize!
            prev_mask = prev_interventions[:, concept_idx] == 1
            prev_interventions[:, concept_idx] = 1
            # See how the predictions change
            _, _, change_y_preds, _, _ = self.cbm(
                x,
                intervention_idxs=prev_interventions,
                c=c,
                latent=latent,
            )
            # Restore prev-interventions
            prev_interventions[:, concept_idx] = prev_mask

            # Compute probability distribution over classes
            if self.n_tasks > 1:
                new_prob_distr = torch.nn.functional.softmax(
                    change_y_preds,
                    dim=-1,
                )
                # And aim to maximize the probability of the ground-truth class
                # assuming we also correctly intervened on the current concept
                expected_change = new_prob_distr[
                    torch.eye(new_prob_distr.shape[-1])[pred_class.cpu()].type(
                        torch.bool
                    )
                ]
            else:
                y_probs = torch.sigmoid(change_y_preds)
                y_probs = torch.squeeze(y_probs, dim=-1)
                expected_change = (
                    y_probs * pred_class + (1 - y_probs) * (1 - pred_class)
                )
        else:
            # Else we actually compute the expectation
            expected_change = torch.zeros(x.shape[0]).to(c.device)
            for concept_val in [0, 1]:
                old_c_vals = c[:, concept_idx]
                # Make a fake intervention in the current concept
                c[:, concept_idx] = concept_val
                prev_mask = prev_interventions[:, concept_idx] == 1
                prev_interventions[:, concept_idx] = 1
                # See how the predictions change
                _, _, change_y_preds, _, _ = self.cbm(
                    x,
                    intervention_idxs=prev_interventions,
                    c=c,
                    latent=latent,
                )
                # Restore prev-interventions
                c[:, concept_idx] = old_c_vals
                prev_interventions[:, concept_idx] = prev_mask
                # And compute their weighted output
                prob = hat_c if concept_val == 1 else 1 - hat_c
                if self.n_tasks > 1:
                    new_prob_distr = torch.nn.functional.softmax(
                        change_y_preds,
                        dim=-1,
                    )

                    expected_change += (
                        prob * new_prob_distr[
                            torch.eye(new_prob_distr.shape[-1])[pred_class.cpu()].type(
                                torch.bool
                            )
                        ]
                    )
                else:
                    new_prob_distr = torch.sigmoid(change_y_preds)
                    new_prob_distr = torch.squeeze(new_prob_distr, dim=-1)
                    expected_change += (
                        new_prob_distr * pred_class +
                        (1 - new_prob_distr) * (1 - pred_class)
                    )
        # Put these together to get the expected change in output probability
        if self._optimal:
            # Then the score will just consider the probability of the ground
            # truth class when we correctly intervened on the current concept
            return expected_change

        # Else we compute the expected change in the predicted probability given
        # an intervention on this concept
        return self._normalize_importance(torch.abs(
            expected_change - pred_class_prob.to(expected_change.device)
        ))

    def _uncertainty_score(
        self,
        concept_idx,
        c,
        pred_c,
    ):
        #  First compute the concept's entropy
        hat_c = pred_c[:, concept_idx]
        if self.use_uncertainty_proxy:
            scores = 1 / torch.abs(hat_c - 0.5 + self.eps)
        else:
            scores = -(
                (hat_c * torch.log(hat_c + self.eps)) +
                (1 - hat_c) * torch.log(1 - hat_c + self.eps)
            )
        return self._normalize_uncertainty(scores)

    def _coop_step(
        self,
        x,
        c,
        pred_c,
        prev_interventions,
        latent,
        y=None,
        y_preds=None,
        competencies=None,
        prior_distribution=None,
    ):
        sample_uncertainties = torch.zeros(c.shape).to(pred_c.device)
        sample_importances = torch.zeros(c.shape).to(pred_c.device)
        if (y is None) or (not self._optimal):
            _, _, y_preds, _, _ = self.cbm(
                x,
                intervention_idxs=prev_interventions,
                c=c,
                latent=latent,
            )
            if self.n_tasks > 1:
                pred_class_prob, pred_class = torch.nn.functional.softmax(
                    y_preds,
                    dim=-1
                ).max(dim=-1)
            else:
                y_probs = torch.sigmoid(y_preds)
                y_probs = torch.squeeze(y_probs, dim=-1)
                pred_class = (y_probs >= 0.5).type(y_preds.type())
                pred_class_prob = (
                    y_probs * pred_class + (1 - y_probs) * (1 - pred_class)
                )
        else:
            # Else we assume we have an oracle that gives us the ground truth
            # concept
            pred_class = y
            pred_class_prob = torch.ones(c.shape[0]).to(pred_c.device)

        if prior_distribution is None:
            denom = torch.sum(prev_interventions, dim=-1, keepdim=True)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            prior_distribution = torch.ones(c.shape).to(c.device) / denom
        elif self.cbm.use_concept_groups and self.include_prior:
            new_prior_distribution = torch.ones_like(pred_c).to(pred_c.device)
            for group_idx, (_, concepts) in enumerate(
                self.concept_group_map.items()
            ):
                for concept_idx in concepts:
                    new_prior_distribution[:, concept_idx] = \
                        prior_distribution[:, group_idx]
            prior_distribution = new_prior_distribution
        for concept_idx in range(c.shape[-1]):
            # If there is at least one element in the batch that has this
            # concept unintervened, then we will have to evaluate its score for
            # all of them
            samples_not_using_concept = prev_interventions[:, concept_idx] == 0
            # If there is at least one element in the batch that has this
            # concept unintervened, then we will have to evaluate its score for
            # all of them
            if np.any(samples_not_using_concept):
                if latent is not None:
                    if not isinstance(latent, tuple):
                        updated_latent = latent[samples_not_using_concept]
                    else:
                        updated_latent = tuple(
                            [x[samples_not_using_concept] for x in latent]
                        )
                x_used = x[samples_not_using_concept, :]
                prev_interventions_used = \
                    prev_interventions[samples_not_using_concept, :]
                c_used = c[samples_not_using_concept, :]
                pred_class_used = pred_class[samples_not_using_concept]
                pred_c_used = pred_c[samples_not_using_concept, :]
                pred_class_prob_used = pred_class_prob[samples_not_using_concept]
                sample_uncertainties[samples_not_using_concept, concept_idx] = \
                    self._uncertainty_score(
                        concept_idx=concept_idx,
                        c=c_used,
                        pred_c=pred_c_used,
                    )
                competencies_used = \
                        competencies[samples_not_using_concept, concept_idx]
                prior_distribution_used = \
                    prior_distribution[samples_not_using_concept, concept_idx]
                sample_importances[samples_not_using_concept, concept_idx] = \
                    self._importance_score(
                        x=x_used,
                        concept_idx=concept_idx,
                        c=c_used,
                        pred_c=pred_c_used,
                        prev_interventions=prev_interventions_used,
                        pred_class=pred_class_used,
                        pred_class_prob=pred_class_prob_used,
                        latent=updated_latent,
                        competencies=competencies_used,
                        prior_distribution=prior_distribution_used,
                    ).type(sample_importances.type())
        # if prior_distribution is not None:
        #     sample_importances *= prior_distribution
        # Updates prev_interventions rather than copying it to speed things up
        if self.group_based:
            # Then we average scores across members of the same group
            group_scores = torch.zeros(
                (c.shape[0], len(self.concept_group_map))
            ).to(
                pred_c.device
            )
            group_names = []
            for group_idx, (group_name, group_concepts) in enumerate(
                self.concept_group_map.items()
            ):
                group_names.append(group_name)
                group_uncertainties = torch.mean(
                    sample_uncertainties[:, group_concepts],
                    dim=-1,
                )
                group_importances = torch.mean(
                    sample_importances[:, group_concepts],
                    dim=-1,
                )
                current_score = (
                    self.concept_entropy_weight * group_uncertainties +
                    self.importance_weight * group_importances
                )
                # Finally include the aquisition cost
                # WLOG, we assume that the CooP multiplier for this cost is
                # merged within the cost itself
                if self.acquisition_costs is not None:
                    # Then we add the cost of aquiring all concepts in the group!
                    current_score += np.sum(
                        self.acquisition_costs[group_concepts]
                    )
                group_scores[:, group_idx] = current_score
            next_groups = torch.argmax(group_scores, axis=-1)
            for sample_idx, best_group_idx in enumerate(next_groups):
                # Get the concepts corresponding to the group we will be
                # intervening on
                next_concepts = \
                    self.concept_group_map[group_names[best_group_idx]]
                prev_interventions[sample_idx, next_concepts] = 1
        else:
            sample_scores = (
                self.concept_entropy_weight * sample_uncertainties +
                self.importance_weight * sample_importances
            )
            # Finally include the aquisition cost
            # WLOG, we assume that the CooP multiplier for this cost is merged
            # within the cost itself
            if self.acquisition_costs is not None:
                sample_scores += self.acquisition_costs

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
            mask = np.zeros((x.shape[0], c.shape[-1]), dtype=np.int64)
        else:
            mask = prev_interventions.detach().cpu().numpy()
        if not self.include_prior:
            prior_distribution = None
        if prior_distribution is None:
            prior_distribution = torch.ones(c.shape).to(c.device) / c.shape[-1]
        elif not self.cbm.use_concept_groups:
            prior_distribution = torch.FloatTensor([
                [
                    torch.sum(prior_distribution[idx, concepts])
                    for _, concepts in self.concept_group_map.items()
                ]
                for idx in range(x.shape[0])
            ]).to(prior_distribution.device)
        _, _, y_preds, _, latent = self.cbm(x)
        if competencies is None:
            competencies = torch.ones(c.shape).to(x.device)
        if self.num_groups_intervened == len(self.concept_group_map):
            return np.ones(c.shape, dtype=np.int64), c
        for i in range(self.num_groups_intervened):
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
            )
        return mask, c

