from torch import nn
import torch.nn.functional as F

from architecture.utils import differentiable_or


class RewardFunction:
    def __init__(self):
        pass

    def get_reward(self, episode):
        pass


class TrueReward:
    def __init__(self):
        pass

    def _get_reward(self, state, instruction):
        pass

    def __call__(self, state, instruction):
        self._get_reward(state, instruction)


class LearnedReward(nn.Module, RewardFunction):
    def __init__(self, context_model, language_model, observation_size, net_params, hidden_layer_size=100):
        super().__init__()
        self.context_net = context_model(**net_params['context_net'], hidden_layer_size=0)

        self.language_model = language_model

        # self.linear_attn = nn.Linear(in_features=self.language_model.out_features, out_features=observation_size)
        # self.hidden_layer = nn.Linear(in_features=observation_size, out_features=hidden_layer_size)
        
        self.reward_layer = nn.Linear(in_features=self.context_net.out_features, out_features=1)

    def forward(self, state, instructions):
        """

        :param state:
        :param instructions: list of string instructions
        :return:
        """
        instruction_embedding = self.language_model(instructions)
        context = self.context_net(state=state, instruction=instruction_embedding)
        reward = self.reward_layer(context)
        # reward = differentiable_or(x.view(-1))
        return reward


if __name__ == "__main__":
    pass
