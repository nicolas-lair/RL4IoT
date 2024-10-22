from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torchtext as text
from sklearn.preprocessing import OneHotEncoder
from logger import get_logger

logger = get_logger(__name__)
logger.setLevel(20)


def get_description_embedder(description_embedder_params, language_model):
    description_embedder_type = description_embedder_params['type']
    if description_embedder_type == 'one_hot':
        description_embedder = OneHotDescriptionEmbedder(**description_embedder_params)
    elif description_embedder_type == 'glove_mean':
        description_embedder = PreTrainedDescriptionEmbedder(**description_embedder_params)
    elif description_embedder_type == 'projection':
        description_embedder = ProjectedDescriptionEmbedder(**description_embedder_params)
    elif description_embedder_type == 'learned_lm':
        assert language_model is not None
        description_embedder = LMBasedDescriptionEmbedder(language_model=language_model, **description_embedder_params)
    else:
        raise NotImplementedError
    return description_embedder


def deal_with_cache(data, cache, func, use_cache=True, **kwargs):
    """
    Identify data that needs to be computed
    Update the cache with the new computed data
    Get computed data from the cache
    Parameters
    ----------
    data :
    cache :
    func :
    use_cache :

    Returns
    -------

    """
    if use_cache:
        new_data = [d for d in data if cache.get(d, None) is None]
    else:
        new_data = data

    computed_data = func(new_data, **kwargs) if new_data else []

    if use_cache:
        new_res = {new_data[i]: computed_data[i] for i in range(len(new_data))}
        cache.update(new_res)
        computed_data = [cache[d] for d in data]
        cache.truncate()

    assert len(computed_data) == len(data)
    return computed_data


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def truncate(self):
        if self._max > 0:
            while len(self) > self._max:
                oldest = next(iter(self))
                del self[oldest]


class OrderedOneHotEncoder:
    def __init__(self, object_list, device, max_size=10):
        assert len(object_list) <= max_size
        self.object_list = object_list.copy()
        self.device = device
        self.max_size = max_size
        self.encoder = dict()
        [self._add_new_encoding(o) for o in object_list]

    def _add_new_encoding(self, obj):
        u = torch.zeros(self.max_size, device=self.device)
        u[self.object_list.index(obj)] = 1
        self.encoder[obj] = u
        return u

    def embed(self, obj):
        encoding = self.encoder.get(obj, None)
        if encoding is None:
            assert len(self.object_list) < self.max_size
            self.object_list.append(obj)
            encoding = self._add_new_encoding(obj)
        return encoding.clone()

    def to(self, device):
        self.device = device
        for o in self.object_list:
            self.encoder[o] = self.encoder[o].to(device)

    def state_dict(self):
        state_dict = {
            k: v.cpu() for k, v in self.encoder.items()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        assert len(state_dict) <= self.max_size
        self.encoder = {
            k: v.to(self.device) for k, v in state_dict.items()
        }


class ItemEmbedder:
    def __init__(self, item_type, device, use_cache=True):
        self.item_type_embedder = OneHotEncoder(sparse=False)
        self.item_type_embedder.fit(np.array(item_type).reshape(-1, 1))
        if use_cache:
            self.cache = {
                item: torch.tensor(self.item_type_embedder.transform(np.array(item).reshape(-1, 1))).to(device).view(-1)
                for item in item_type
            }
        else:
            self.cache = None

    def embed(self, item):
        return self.cache[item]


class StateEmbedder:
    def __init__(self, description_embedder, item_type, value_encoding_size, device, use_cache=True):
        self.description_embedder = description_embedder
        self.item_embedder = ItemEmbedder(item_type=item_type, device=device)
        self.value_embedding_size = value_encoding_size

        self.device = device
        self.description_embedder.to(self.device)

        self.use_cache = use_cache
        # if self.use_cache:
        #     self.channel_cache = FixSizeOrderedDict(max=10000)

        self.always_use_cache = self.use_cache and not isinstance(self.description_embedder, LearnedDescriptionEmbedder)

    def _compute_description_embedding_in_batch(self, observations, use_description_cache):
        """

        Parameters
        ----------
        observations : list of observations
        use_description_cache : boolean

        Returns
        -------

        """
        thing_descriptions, channels_descriptions = [], []
        for obs in observations:
            thing_descriptions.extend([t['description'] for t in obs.values()])
            channels_descriptions.extend(
                [c['description'] for t in obs.values() for c in t.values() if isinstance(c, dict)])

        thing_descriptions_embedding = self.description_embedder.embed_descriptions(thing_descriptions,
                                                                                    use_cache=use_description_cache,
                                                                                    type='thing')
        channel_descriptions_embedding = self.description_embedder.embed_descriptions(channels_descriptions,
                                                                                      use_cache=use_description_cache,
                                                                                      type='channel')
        # descriptions_embeddings = self.description_embedder.embed_descriptions(
        #     thing_descriptions + channels_descriptions, use_cache=use_description_cache)
        #
        # thing_description_index = 0
        # channel_description_index = len(thing_descriptions)
        # return descriptions_embeddings, (thing_description_index, channel_description_index)
        return thing_descriptions_embedding, channel_descriptions_embedding

    def _build_channel_embedding(self, channel):
        item_embedding = self.item_embedder.embed(channel['item_type'])

        # Value embedding
        value_embedding = torch.zeros(self.value_embedding_size).to(self.device)
        if isinstance(channel['state'][0], str):
            value_embedding[:len(channel['state'])] = torch.tensor([int(channel['state'][0])])
        else:
            value_embedding[:len(channel['state'])] = torch.tensor(channel['state'])
        return item_embedding, value_embedding

    def _compute_state_embedding(self, observations, use_cache=True):
        """

        Parameters
        ----------
        observations : list of observation (State object)
        use_cache : boolean

        Returns
        -------

        """
        # descr_embeddings, (thing_index, channel_index) = self._compute_description_embedding_in_batch(
        #     observations=observations, use_description_cache=use_cache)
        thingdesc_embedding, channeldesc_embeding = self._compute_description_embedding_in_batch(
            observations=observations, use_description_cache=use_cache)

        thing_index, channel_index = 0, 0
        embedded_observations = []
        for obs in observations:
            new_obs = dict()
            for thing_name, thing_state in obs.items():
                thing_obs = []
                thing_description_embedding = thingdesc_embedding[thing_index].view(-1)
                thing_index += 1
                for channel_state in thing_state.get_channels_state():
                    if channel_state['item_type'] == 'goal_string':
                        raise NotImplementedError
                    channel_descriptions_embedding = channeldesc_embeding[channel_index].view(-1)
                    channel_index += 1

                    channel_embedding = None
                    # if use_cache:
                    #     channel_embedding = self.channel_cache.get(channel, None)
                    if not use_cache or channel_embedding is None:
                        item_embedding, value_embedding = self._build_channel_embedding(channel_state)
                        channel_embedding = torch.cat(
                            [channel_descriptions_embedding, item_embedding, value_embedding,
                             thing_description_embedding]).to(self.device).float()
                        # if use_cache:
                        #     self.channel_cache[channel] = channel_embedding
                    thing_obs.append(channel_embedding)

                new_obs[thing_name] = torch.stack(thing_obs)
                assert len(new_obs[thing_name].size()) == 2
            embedded_observations.append(new_obs)
        return embedded_observations

    def embed_state(self, observations, use_cache=None):
        if not isinstance(observations, (tuple, list)):
            observations = [observations]

        if self.always_use_cache:
            if not use_cache:
                logger.debug('State embedder using the cache due to always_use_cache attr. while use_cache is False')
            use_cache = True
        else:
            use_cache = self.use_cache if use_cache is None else use_cache

        # TODO deal with cache size
        observation_embedding = self._compute_state_embedding(observations, use_cache=use_cache)
        # observation_embedding = deal_with_cache(observations, self.cache, self._compute_state_embedding,
        #                                         use_cache=use_cache)

        if len(observation_embedding) == 1:
            observation_embedding = observation_embedding[0]
        return observation_embedding

    def empty_cache(self):
        if isinstance(self.description_embedder, LearnedDescriptionEmbedder):
            # self.cache.clear()
            self.description_embedder.clear_cache()


class AbstractDescriptionEmbedder(ABC):
    cache = FixSizeOrderedDict(max=2000)

    @abstractmethod
    def embed_descriptions(self, descriptions, type, use_cache=True):
        pass

    def clear_cache(self):
        self.cache.clear()

    def eval(self):
        pass

    def train(self):
        pass

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict):
        raise NotImplementedError


class OneHotDescriptionEmbedder(AbstractDescriptionEmbedder):
    def __init__(self, thing_list, channel_list, max_thing, max_channel,
                 device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        self.device = device
        self.encoder = {
            'thing': OrderedOneHotEncoder(object_list=thing_list, device=device, max_size=max_thing),
            'channel': OrderedOneHotEncoder(object_list=channel_list, device=device, max_size=max_channel)
        }

    def embed_descriptions(self, descriptions, type, use_cache=True):
        if isinstance(descriptions, str):
            return self.encoder[type].embed(descriptions)
        elif isinstance(descriptions, list):
            return [self.encoder[type].embed(d) for d in descriptions]
        else:
            raise NotImplementedError

    def to(self, device):
        self.device = device
        [encoder.to(device) for encoder in self.encoder.values()]

    def state_dict(self):
        return dict(thing_encoder=self.encoder['thing'].state_dict(),
                    channel_encoder=self.encoder['channel'].state_dict())

    def load_state_dict(self, state_dict):
        for k, v in self.encoder.items():
            v.load_state_dict(state_dict[f'{k}_encoder'])


class PreTrainedDescriptionEmbedder(AbstractDescriptionEmbedder):
    def __init__(self, vocab, reduction='mean', device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        self.tokenizer = text.data.utils.get_tokenizer(tokenizer="spacy", language="en")
        self.lower_case = True
        self.pytorch_device = device
        self.vocab = vocab

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
        self.cache.update({description: description_embedding})
        return description_embedding

    def embed_descriptions(self, descriptions, type, **kwargs):
        def aux(d):
            assert isinstance(d, str), 'description should be goal_string'
            embedded_description = self.cache.get(d, self.embed_single_description(d))
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

    def clear_cache(self, force=False):
        if force:
            super().clear_cache()


class LearnedDescriptionEmbedder(AbstractDescriptionEmbedder):
    @abstractmethod
    def compute_batch_description_embedding(self, descriptions):
        pass

    def embed_descriptions(self, descriptions, type, use_cache=True):
        if isinstance(descriptions, str):
            descriptions = [descriptions]

        description_embedding = deal_with_cache(descriptions, self.cache, self.compute_batch_description_embedding,
                                                use_cache=use_cache)
        if isinstance(description_embedding, list):
            description_embedding = torch.stack(description_embedding)
        if len(description_embedding) == 1:
            description_embedding.squeeze()
        return description_embedding


class ProjectedDescriptionEmbedder(nn.Module, LearnedDescriptionEmbedder):
    def __init__(self, vocab, embedding_size, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        super().__init__()
        self.tokenizer = text.data.utils.get_tokenizer(tokenizer="spacy", language="en")
        self.lower_case = True
        self.vocab = vocab
        self.device = device

        self.projection_layer = nn.Sequential(
            nn.Linear(in_features=vocab.dim, out_features=embedding_size, bias=False),
            nn.Sigmoid()
        )

    def _get_one_embedding(self, description):
        tokens = self.tokenizer(description)
        embedding = self.vocab.get_vecs_by_tokens(tokens, lower_case_backup=self.lower_case)
        return embedding

    def compute_batch_description_embedding(self, descriptions):
        word_embeddings = pad_sequence([self._get_one_embedding(d) for d in descriptions], batch_first=True)
        description_embedding = self.projection_layer(word_embeddings.to(self.device))
        description_embedding = description_embedding.mean(dim=1)
        description_embedding = description_embedding  # squeeze dimension for single descriptions
        return description_embedding

    def to(self, device):
        super(ProjectedDescriptionEmbedder, self).to(device)
        self.device = device


class LMBasedDescriptionEmbedder(nn.Module, LearnedDescriptionEmbedder):
    def __init__(self, language_model, embedding_size, **kwargs):
        super().__init__()
        self.language_model = language_model
        self.projection_layer = nn.Sequential(
            nn.Linear(in_features=language_model.out_features, out_features=embedding_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, descriptions):
        x = self.language_model(descriptions)
        x = self.projection_layer(x)
        return x

    def compute_batch_description_embedding(self, descriptions):
        return self.forward(descriptions)
