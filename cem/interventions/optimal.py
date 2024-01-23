import numpy as np
import itertools
import torch
from .coop import CooP
import logging
from cem.interventions.intervention_policy import InterventionPolicy

class GreedyOptimal(CooP):
    def __init__(
        self,
        concept_group_map,
        cbm,
        n_tasks,
        num_groups_intervened=0,
        acquisition_weight=1,
        importance_weight=1,
        acquisition_costs=None,
        group_based=True,
        eps=1e-8,
        include_prior=True,
        **kwargs
    ):
        CooP.__init__(
            self,
            num_groups_intervened=num_groups_intervened,
            concept_group_map=concept_group_map,
            cbm=cbm,
            concept_entropy_weight=0,
            importance_weight=importance_weight,
            acquisition_weight=acquisition_weight,
            acquisition_costs=acquisition_costs,
            group_based=group_based,
            eps=eps,
            include_prior=include_prior,
            n_tasks=n_tasks,
            **kwargs,
        )
        self._optimal = True

class TrueOptimal(InterventionPolicy):
    def __init__(
        self,
        concept_group_map,
        cbm,
        n_tasks,
        num_groups_intervened=0,
        acquisition_costs=None,
        acquisition_weight=1,
        importance_weight=1,
        group_based=True,
        eps=1e-8,
        include_prior=True,
        **kwargs
    ):
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.acquisition_costs = acquisition_costs
        self.group_based = group_based
        self._optimal = True
        self.cbm = cbm
        self.n_tasks = n_tasks
        self.eps = eps
        self.acquisition_weight = acquisition_weight
        self.importance_weight = importance_weight
        self.include_prior = include_prior
        self.debug_count = 0

    def _importance_scores(
        self,
        x,
        c,
        y,
        concepts_to_intervene,
        latent,
    ):
        # Then just look at the value of the probability of the known truth
        # class, as this is what we want to maximize!
        # See how the predictions change
        _, _, y_pred_logits, _, _ = self.cbm(
            x,
            intervention_idxs=concepts_to_intervene,
            c=c,
            latent=latent,
        )
        y_pred_logits = y_pred_logits.detach().cpu()

        # if (self.count == 0):
        #     logging.debug(
        #         f"num_groups_intervened: {self.num_groups_intervened}\n\n\n"
        #         f"concept_group_map: {self.concept_group_map}\n\n\n"
        #         f"acquisition_costs: {self.acquisition_costs}\n\n\n"
        #         f"n_tasks: {self.n_tasks}\n\n\n"
        #         f"eps: {self.eps}\n\n\n"
        #         f"acquisition_weight: {self.acquisition_weight}\n\n\n"
        #         f"importance_weight: {self.importance_weight}\n\n\n"
        #         f"x: {x}\n\n\n"
        #         f"x.shape: {x.shape}\n\n\n"
        #         f"intervention_idxs: {concepts_to_intervene}\n\n\n"
        #         f"y_pred_logits: {y_pred_logits}\n\n\n`"
        #         f"y_pred_logits.shape: {y_pred_logits.shape}\n\n\n"
        #     )

        if self.n_tasks <= 1:
            y_probs = torch.sigmoid(y_pred_logits)
            ones_tensor = torch.ones_like(y_probs)
            ones_tensor -= y_probs
            y_pred_logits = torch.cat((ones_tensor, y_probs), dim = 1)

        ret = np.array([
            y_pred_logits[i, label.int()].numpy()
            for i, label in enumerate(y.clone().detach().cpu())
        ])
        
        # if (self.count == 0):
        #     logging.debug(
        #         f"updated y_pred_logits: {y_pred_logits}\n\n\n"
        #         f"updated y_pred_logits.shape: {y_pred_logits.shape}\n\n\n"
        #         f"ret: {ret}\n\n\n"
        #         f"ret.shape: {ret.shape}\n\n\n"
        #     )
        #     self.count = 1

        return ret


    def _opt_score(
        self,
        x,
        c,
        y,
        concepts_to_intervene,
        latent,
    ):
        #  First compute the test accuracy for this intervention set
        importance_scores = self._importance_scores(
            x=x,
            c=c,
            y=y,
            concepts_to_intervene=concepts_to_intervene,
            latent=latent,
        )
        scores = self.importance_weight * importance_scores

        # Finally include the aquisition cost
        if self.acquisition_costs is not None:
            scores += self.acquisition_costs * self.acquisition_weight
        return scores
    
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
        if self.num_groups_intervened == 0:
            return mask, c
        _, _, _, _, latent = self.cbm(x)
        scores = []
        intervened_concepts = []
        concept_group_names = list(self.concept_group_map.keys())

        for intervention_idxs in itertools.combinations(
            set(range(len(concept_group_names))),
            self.num_groups_intervened,
        ):
            real_intervention_idxs = []
            for group_idx in intervention_idxs:
                real_intervention_idxs.extend(
                    self.concept_group_map[concept_group_names[group_idx]]
                )
            intervention_idxs = sorted(real_intervention_idxs)
            intervened_concepts.append(intervention_idxs)
            current_scores = self._opt_score(
                x=x,
                c=c,
                y=y,
                concepts_to_intervene=intervention_idxs,
                latent=latent,
            )
            scores.append(np.expand_dims(current_scores, axis=-1))
        scores = np.concatenate(scores, axis=-1)
        best_scores = np.argmax(scores, axis=-1)
        mask = np.zeros(c.shape, dtype=np.int32)
        logging.debug(
            f"Printing info for optimal intervention policy:\n"
            f"intervened_concepts: {intervened_concepts}\n\n\n"
            f"scores: {scores}\n"
            f"scores.shape: {scores.shape}\n\n\n"
            f"best_scores: {best_scores}\n"
            f"best_scores.shape: {best_scores.shape}\n\n\n"
        )

        # if (self.count == 1):
        #     for sample_idx in range(10, x.shape[0]):
        #         best_score_idx = best_scores[sample_idx]
        #         # Set the concepts of the best-scored model to be intervened
        #         # for this sample
        #         curr_mask = np.zeros((c.shape[-1],), dtype=np.int32)
        #         for idx in intervened_concepts[best_score_idx]:
        #             mask[sample_idx, idx] = 1
        #         logging.debug(
        #             f"Printing info for selected best index:\n"
        #             f"selected concepts with best scores: {best_score_idx}\n\n\n"
        #             f"Scores: {scores[sample_idx]}\n\n\n"
        #         )
        #     self.count += 1
            
        for sample_idx in range(x.shape[0]):
            best_score_idx = best_scores[sample_idx]
            # Set the concepts of the best-scored model to be intervened
            # for this sample
            curr_mask = np.zeros((c.shape[-1],), dtype=np.int32)
            for idx in intervened_concepts[best_score_idx]:
                mask[sample_idx, idx] = 1
        return mask, c

class TrueOptimalFix(InterventionPolicy):
    def __init__(
        self,
        concept_group_map,
        cbm,
        n_tasks,
        num_groups_intervened=0,
        acquisition_costs=None,
        acquisition_weight=1,
        importance_weight=1,
        group_based=True,
        eps=1e-8,
        include_prior=True,
        **kwargs
    ):
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.acquisition_costs = acquisition_costs
        self.group_based = group_based
        self._optimal = True
        self.cbm = cbm
        self.n_tasks = n_tasks
        self.eps = eps
        self.acquisition_weight = acquisition_weight
        self.importance_weight = importance_weight
        self.include_prior = include_prior
        self.debug_count = 0

    def _importance_scores(
        self,
        x,
        c,
        y,
        concepts_to_intervene,
        latent,
    ):
        # Then just look at the value of the probability of the known truth
        # class, as this is what we want to maximize!
        # See how the predictions change
        _, _, y_pred_logits, _, _ = self.cbm(
            x,
            intervention_idxs=concepts_to_intervene,
            c=c,
            latent=latent,
        )
        y_pred_logits = y_pred_logits.detach().cpu()

        # if (self.count == 0):
        #     logging.debug(
        #         f"num_groups_intervened: {self.num_groups_intervened}\n\n\n"
        #         f"concept_group_map: {self.concept_group_map}\n\n\n"
        #         f"acquisition_costs: {self.acquisition_costs}\n\n\n"
        #         f"n_tasks: {self.n_tasks}\n\n\n"
        #         f"eps: {self.eps}\n\n\n"
        #         f"acquisition_weight: {self.acquisition_weight}\n\n\n"
        #         f"importance_weight: {self.importance_weight}\n\n\n"
        #         f"x: {x}\n\n\n"
        #         f"x.shape: {x.shape}\n\n\n"
        #         f"intervention_idxs: {concepts_to_intervene}\n\n\n"
        #         f"y_pred_logits: {y_pred_logits}\n\n\n`"
        #         f"y_pred_logits.shape: {y_pred_logits.shape}\n\n\n"
        #     )

        if self.n_tasks <= 1:
            y_probs = torch.sigmoid(y_pred_logits)
            ones_tensor = torch.ones_like(y_probs)
            ones_tensor -= y_probs
            y_pred_logits = torch.cat((ones_tensor, y_probs), dim = 1)

        ret = np.array([
            y_pred_logits[i, label.int()].numpy()
            for i, label in enumerate(y.clone().detach().cpu())
        ])
        
        # if (self.count == 0):
        #     logging.debug(
        #         f"updated y_pred_logits: {y_pred_logits}\n\n\n"
        #         f"updated y_pred_logits.shape: {y_pred_logits.shape}\n\n\n"
        #         f"ret: {ret}\n\n\n"
        #         f"ret.shape: {ret.shape}\n\n\n"
        #     )
        #     self.count = 1

        return ret


    def _opt_score(
        self,
        x,
        c,
        y,
        concepts_to_intervene,
        latent,
    ):
        #  First compute the test accuracy for this intervention set
        importance_scores = self._importance_scores(
            x=x,
            c=c,
            y=y,
            concepts_to_intervene=concepts_to_intervene,
            latent=latent,
        )
        scores = self.importance_weight * importance_scores

        # Finally include the aquisition cost
        if self.acquisition_costs is not None:
            scores += self.acquisition_costs * self.acquisition_weight
        return scores
    
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
        if self.num_groups_intervened == 0:
            return mask, c
        _, _, _, _, latent = self.cbm(x)
        concept_group_names = list(self.concept_group_map.keys())
        intervention_combinations = list(itertools.combinations(
            set(range(len(concept_group_names))),
            self.num_groups_intervened,
        ))
        # Shape = (B, Combinations)
        scores = np.zeros((x.shape[0], len(intervention_combinations)), dtype=np.int64)

        intervened_concepts = []
        # for intervention_idxs in itertools.combinations(
        #     set(range(len(concept_group_names))),
        #     self.num_groups_intervened,
        # ):
        #     real_intervention_idxs = []
        #     for group_idx in intervention_idxs:
        #         real_intervention_idxs.extend(
        #             sorted(self.concept_group_map[concept_group_names[group_idx]])
        #         )
        #     intervention_idxs = real_intervention_idxs
        #     intervened_concepts.append(intervention_idxs)
        #     current_scores = self._opt_score(
        #         x=x,
        #         c=c,
        #         y=y,
        #         concepts_to_intervene=intervention_idxs,
        #         latent=latent,
        #     )
        #     scores.append(np.expand_dims(current_scores, axis=-1))

        for idx in range(len(intervention_combinations)):
            intervention_idxs = intervention_combinations[idx]
            real_intervention_idxs = []
            for group_idx in intervention_idxs:
                real_intervention_idxs.extend(
                    sorted(self.concept_group_map[concept_group_names[group_idx]])
                )
            intervention_idxs = real_intervention_idxs
            # intervened_concepts.append(intervention_idxs)
            current_scores = self._opt_score(
                x=x,
                c=c,
                y=y,
                concepts_to_intervene=intervention_idxs,
                latent=latent,
            )
            
            if self.debug_count == 0:
                logging.debug(
                    f"Overall score shape: {scores.shape}\n"
                    f"Score for combination {intervention_idxs}\n"
                    f"current_scores: {current_scores}"
                )
            for score_idx in range(len(current_scores)):
                scores[score_idx][idx] = current_scores[score_idx]

        best_score_idxs = np.argmax(scores, axis=-1)
        mask = np.zeros(c.shape, dtype=np.int32)
        # logging.debug(
        #     f"Printing info for optimal intervention policy:\n"
        #     f"intervened_concepts: {intervened_concepts}\n\n\n"
        #     f"scores: {scores}\n"
        #     f"scores.shape: {scores.shape}\n\n\n"
        #     f"best_scores: {best_scores_idx}\n"
        #     f"best_scores.shape: {best_scores_idx.shape}\n\n\n"
        # )

        # if (self.count == 1):
        #     for sample_idx in range(10, x.shape[0]):
        #         best_score_idx = best_scores[sample_idx]
        #         # Set the concepts of the best-scored model to be intervened
        #         # for this sample
        #         curr_mask = np.zeros((c.shape[-1],), dtype=np.int32)
        #         for idx in intervened_concepts[best_score_idx]:
        #             mask[sample_idx, idx] = 1
        #         logging.debug(
        #             f"Printing info for selected best index:\n"
        #             f"selected concepts with best scores: {best_score_idx}\n\n\n"
        #             f"Scores: {scores[sample_idx]}\n\n\n"
        #         )
        #     self.count += 1

        if self.debug_count == 0:
            logging.debug(
                f"scores: {scores}"
                f"best_scores: {best_score_idxs}\n"
                f"intervened_concepts: {intervened_concepts}\n"
                f"concept_group_names: {concept_group_names}\n"
            )
            self.debug_count += 1
            
        for sample_idx in range(x.shape[0]):
            best_score_idx = best_score_idxs[sample_idx]
            # Set the concepts of the best-scored model to be intervened
            # for this sample
            curr_mask = np.zeros((c.shape[-1],), dtype=np.int32)
            for idx in intervention_combinations[best_score_idx]:
                mask[sample_idx, idx] = 1
        return mask, c
