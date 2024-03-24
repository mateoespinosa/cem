from .coop import CooP

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