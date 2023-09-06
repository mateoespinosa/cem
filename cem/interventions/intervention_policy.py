from abc import ABC, abstractmethod

class InterventionPolicy(ABC):

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
        self.cbm = cbm
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.group_based = group_based
        self.include_prior = include_prior
        self.horizon = horizon
        self.cbm_use_concept_groups = cbm.use_concept_groups


    @abstractmethod
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
        raise NotImplementedError("This is an abstract method!")