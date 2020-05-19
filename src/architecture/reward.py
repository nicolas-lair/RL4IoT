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
    def __init__(self, language_model, observation_size, hidden_layer_size=100):
        super().__init__()

        self.language_model = language_model
        self.linear_attn = nn.Linear(in_features=self.language_model.out_features, out_features=observation_size)

        # self.hidden_layer = nn.Linear(in_features=observation_size, out_features=hidden_layer_size)
        self.reward_layer = nn.Linear(in_features=hidden_layer_size, out_features=1)

    def forward(self, state, instruction):
        instruction_embedding = self.language_model(instruction)
        attn_vector = self.linear_attn(instruction_embedding)
        attn_vector = F.sigmoid(attn_vector)

        x = attn_vector * state
        x = self.reward_layer(x)
        x = F.sigmoid(x)

        reward = differentiable_or(x.view(-1))
        return reward


if __name__ == "__main__":
    pass
