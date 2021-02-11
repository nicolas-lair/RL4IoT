from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import nn
import numpy as np
import torchtext as text
from sklearn.preprocessing import OneHotEncoder


def get_description_embedder(description_embedder_params, language_model):
    description_embedder_type = description_embedder_params['type']
    if description_embedder_type == 'glove_mean':
        description_embedder = PreTrainedDescriptionEmbedder(**description_embedder_params)
    elif description_embedder_type == 'learned_lm':
        assert language_model is not None
        description_embedder = LearnedDescriptionEmbedder(language_model=language_model, **description_embedder_params)
    else:
        raise NotImplementedError
    return description_embedder


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)


class StateEmbedder:
    def __init__(self, description_embedder, item_type, value_encoding_size, device, use_cache):
        self.description_embedder = description_embedder
        self.item_type_embedder = OneHotEncoder(sparse=False)
        self.item_type_embedder.fit(np.array(item_type).reshape(-1, 1))
        self.value_embedding_size = value_encoding_size

        self.device = device
        self.description_embedder.to(self.device)

        self.use_cache = use_cache
        if isinstance(description_embedder, PreTrainedDescriptionEmbedder):
            self.use_cache = True

        if self.use_cache:
            self.cache = FixSizeOrderedDict(max=5000)

        self.cached_state = None
        self.cached_embedding = None

    def preprocess_raw_observation(self, observation):
        if self.use_cache:
            try:
                return self.cache[observation.id]
            except KeyError:
                pass

        new_obs = dict()
        for thing_name, thing in observation.items():
            thing_obs = []
            thing_description = thing['description']
            thing_description_embedding = self.description_embedder.embed_descriptions(thing_description)
            for channel_name, channel in thing.items():
                if channel_name == 'description':
                    break
                if channel['item_type'] == 'goal_string':
                    raise NotImplementedError

                description_embedding = self.description_embedder.embed_descriptions(channel['description'])

                # Item embedding
                item_embedding = self.item_type_embedder.transform(
                    np.array(channel['item_type']).reshape(-1, 1)).flatten()
                item_embedding = torch.tensor(item_embedding).to(self.device)

                # Value embedding
                value_embedding = torch.zeros(self.value_embedding_size).to(self.device)
                value_embedding[:len(channel['state'])] = torch.tensor(channel['state'])

                channel_embedding = torch.cat(
                    [description_embedding.view(-1), item_embedding, value_embedding,
                     thing_description_embedding.view(-1)])
                # if self.pytorch:
                #     channel_embedding = torch.tensor(channel_embedding)
                thing_obs.append(channel_embedding.to(self.device))
            new_obs[thing_name] = torch.stack(thing_obs).float()

        if self.use_cache:
            self.cache[observation.id] = new_obs
        return new_obs

    def embed_state(self, state):
        if isinstance(state, (tuple, list)):
            states_embedding = [self.preprocess_raw_observation(obs) for obs in state]
            return states_embedding
        else:
            if self.cached_embedding and state == self.cached_state:
                pass
            else:
                self.cached_state = state
                self.cached_embedding = self.preprocess_raw_observation(state)
            assert self.cached_embedding is not None
            return self.cached_embedding

    def empty_cache(self):
        if self.use_cache:
            self.cache.clear()

class AbstractDescriptionEmbedder(ABC):
    @abstractmethod
    def embed_descriptions(self, descriptions):
        pass


class PreTrainedDescriptionEmbedder(AbstractDescriptionEmbedder):
    def __init__(self, embedding, word_embedding_params, reduction='mean', authorize_cache=True, **kwargs):
        self.authorize_cache = authorize_cache
        self.cached_embedded_description = dict()
        self.tokenizer = text.data.utils.get_tokenizer(tokenizer="spacy", language="en")
        self.lower_case = True
        self.pytorch_device = 'cpu'
        if embedding == 'glove':
            self.vocab = text.vocab.GloVe(**word_embedding_params)
        else:
            raise NotImplementedError

        if reduction == 'mean':
            self.reduction_func = lambda x: torch.mean(x, dim=0)
        elif reduction == "no_reduction":
            self.reduction_func = lambda x: x
        else:
            raise NotImplementedError

    def embed_single_description(self, description):
        assert isinstance(description, str), 'description should be goal_string'
        tokens = self.tokenizer(description)
        description_embedding = self.vocab.get_vecs_by_tokens(tokens, lower_case_backup=self.lower_case)
        description_embedding = self.reduction_func(description_embedding)
        if self.authorize_cache:
            self.cached_embedded_description.update({description: description_embedding})
        return description_embedding

    def embed_descriptions(self, descriptions, use_cache=True):
        def aux(d):
            assert isinstance(d, str), 'description should be goal_string'
            if use_cache:
                embedded_description = self.cached_embedded_description.get(d, self.embed_single_description(d))
            else:
                embedded_description = self.embed_single_description(d)
            return embedded_description.to(self.pytorch_device)

        if isinstance(descriptions, str):
            embedding = aux(descriptions)
        elif isinstance(descriptions, list):
            embedding = []
            for d in descriptions:
                embedding.append(aux(d))
        else:
            raise TypeError('descriptions should be provided as list or str')

        return embedding

    def to(self, device):
        self.pytorch_device = device


class LearnedDescriptionEmbedder(nn.Module):
    def __init__(self, language_model, embedding_size, **kwargs):
        super().__init__()
        self.language_model = language_model
        self.projection_layer = nn.Sequential(
            nn.Linear(in_features=language_model.out_features, out_features=embedding_size),
            nn.Sigmoid()
        )

    def forward(self, descriptions):
        x = self.language_model(descriptions)
        x = self.projection_layer(x)
        return x

    def embed_descriptions(self, descriptions):
        return self.forward(descriptions)
