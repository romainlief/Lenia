from addict import Dict


class RunDataEntry(Dict):
    """
    Class that specify for RunData entry in the DB
    """

    def __init__(self, db, id, policy_parameters, observations, **kwargs):
        """
        :param kwargs: flexible structure of the entry which might contain additional columns (eg: source_policy_idx, target_goal, etc.)
        """
        super().__init__(**kwargs)
        self.db = db
        self.id = id
        self.policy_parameters = policy_parameters
        self.observations = observations
