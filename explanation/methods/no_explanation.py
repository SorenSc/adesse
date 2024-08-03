from explanation.explainer import RepositioningAgentExplainer


class NoExplanation(RepositioningAgentExplainer):

    def __init__(self):
        self.name = 'NoExplanation'

    def explain(self):
        return None
