from .adversarial_environment import AdversarialEnv


def make(environment_id, opponent_policy=None):
    return AdversarialEnv(environment_id, opponent_policy)
