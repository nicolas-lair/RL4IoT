class RewardFunction:
    def __init__(self, model, language_model, params):
        self.device = params['device']
        self.reward_function = model(**params['model_params'])
        self.reward_function.to(self.device)

        self.language_model = language_model.to(self.device)
        self.language_model.device = self.device

    def train(self, dataset):


