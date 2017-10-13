from .adversarial_environment import AdversarialEnv


def make(environment_id, opponent_policy):
    return AdversarialEnv(environment_id, opponent_policy)
