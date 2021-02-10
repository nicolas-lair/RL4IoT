import torch
import torch.nn as nn
import torch.nn.functional as F

from action_embedder import ActionModel
from architecture.instruction_pointer_network import InstructedPointerNetwork


class BasicQnet(nn.Module):
    def __init__(self, in_features, qnet_params):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=qnet_params['hidden1_out']),
            nn.ReLU(),
            nn.Linear(in_features=qnet_params['hidden1_out'], out_features=qnet_params['hidden2_out']),
            nn.ReLU(),
            nn.Linear(in_features=qnet_params['hidden2_out'], out_features=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.q_network(x)


class FullNet(nn.Module):
    def __init__(self, context_model, action_embedding_size, net_params, raw_action_size, discrete_params):
        super().__init__()

        # self.action_projector = ActionProjector(in_features=max(raw_action_size.values()),
        #                                         out_features=action_embedding_size,
        #                                         env_discrete_params=discrete_params)

        self.context_net = context_model(**net_params['context_net'])

        self.qnet_in_features = self.context_net.out_features + action_embedding_size
        self.q_network = BasicQnet(self.qnet_in_features, qnet_params=net_params['q_network'])

    def compute_context(self, instruction, state, actions, hidden_state):
        context = self.context_net(state=state, instruction=instruction, hidden_state=hidden_state)
        context = context.unsqueeze(1).repeat_interleave(repeats=actions.size(1), dim=1)
        context = torch.cat([context, actions], dim=2)
        return context

    def forward(self, instruction, state, actions, hidden_state):
        """

        Parameters
        ----------
        instruction : torch tensor (batch_size, instruction_embedding)
        state : dict or list of dict
        actions : tuple(List(embedded actions), List(action_type))
        hidden_state : torch tensor

        Returns
        -------
        x: Q_value torch tensor batch size, max_action, 1)
        embedded_actions : torch tensor containing the projection of the embedded actions in a common action space
        """
        # embedded_actions = self.action_projector(*actions)
        context = self.compute_context(instruction, state, actions, hidden_state)
        x = self.q_network(context)
        return x


class FullNetWithAttention(FullNet):
    def __init__(self, context_model, action_embedding_size, net_params, raw_action_size, discrete_params):
        super().__init__(context_model, action_embedding_size, net_params, raw_action_size, discrete_params)
        self.attention_layer = nn.Sequential(
            nn.Linear(in_features=net_params['context_net']['instruction_embedding'],
                      out_features=self.qnet_in_features
                      ),
            nn.Sigmoid()
        )

    def forward(self, instruction, state, actions, hidden_state):
        context = self.compute_context(instruction, state, actions, hidden_state)
        attention_vector = self.attention_layer(instruction)
        x = self.q_network(attention_vector * context)
        return x


class DoubleDeepSetQnet(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, action_embedding_size, net_params, raw_action_size,
                 discrete_params):
        super().__init__()
        self.in_features = net_params['scaler_layer']['latent_out'] + action_embedding_size

        self.action_projector = ActionModel(raw_action_size=max(raw_action_size.values()),
                                            out_features=action_embedding_size,
                                            env_discrete_params=discrete_params)
        self.state_attention_layer = nn.Sequential(
            nn.Linear(instruction_embedding, state_embedding + action_embedding_size),
            nn.Sigmoid()
        )
        self.scaler_layer = nn.Sequential(
            nn.Linear(state_embedding + action_embedding_size, net_params['scaler_layer']['hidden1_out']),
            nn.ReLU(),
            nn.Linear(net_params['scaler_layer']['hidden1_out'], net_params['scaler_layer']['latent_out']),
            nn.ReLU()
        )

        self.q_network = BasicQnet(self.in_features, qnet_params=net_params['q_network'])


class DeepSetActionCritic(nn.Module):
    def __init__(self, instruction_embedding, state_embedding, action_embedding, n_step, net_params):
        super().__init__()
        self.action_projector = nn.Linear(in_features=action_embedding, out_features=net_params['action_projection'],
                                          bias=False)
        self.in_features = instruction_embedding + state_embedding + action_embedding + n_step + net_params[
            'action_projection']
        self.hidden1 = nn.Linear(in_features=self.in_features, out_features=net_params['hidden1_out'])
        self.hidden2 = nn.Linear(in_features=net_params['hidden2_out'], out_features=net_params['hidden2_out'])
        self.final_layer = nn.Linear(in_features=net_params['hidden2_out'], out_features=1)

    # TODO check with gated attention on instruction
    def forward(self, instruction, state, previous_action, actions, step):
        projected_action = self.action_projector(actions)
        action_context = projected_action.sum(dim=0)

        context = torch.cat([instruction, state, previous_action, step, action_context])
        c = context.repeat_interleave(actions.size(0), dim=0)

        x = torch.cat([c, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.final_layer(x)
        x = x.view(-1)
        action_distribution = F.softmax(x)
        return action_distribution


class AttentionCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, instruction, state, previous_action, actions, step):
        pass


class SelfAttentionCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, instruction, state, previous_action, actions, step):
        pass


class PointerCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, instruction, state, previous_action, actions, step):
        pass


class DifferentPtrNetCritic(nn.Module):
    def __init__(self, language_model, thing_selector_params, channel_selector_params, action_selector_params):
        super().__init__()
        self.language_model = language_model
        self.thing_selector = InstructedPointerNetwork(**thing_selector_params)
        self.channel_selector = InstructedPointerNetwork(**channel_selector_params)
        self.action_selector = InstructedPointerNetwork(**action_selector_params)

    def forward(self, instruction, state, previous_action, actions, step):
        embedded_instruction = self.language_model(instruction)
        condition = torch.cat([embedded_instruction, state, previous_action], dim=1)

        # TODO use action sampled from action_proba rather than argmax for better exploration during learning phase
        if step == 'thing':
            action_proba, best_action = self.thing_selector(sequence=actions, condition_embedding=condition)
        elif step == "channel":
            action_proba, best_action = self.channel_selector(sequence=actions, condition_embedding=condition)
        elif step == "action":
            action_proba, best_action = self.action_selector(sequence=actions, condition_embedding=condition)
        elif step == "params":
            selector = None
        elif step == "go":
            selector = None
        else:
            raise NotImplementedError
