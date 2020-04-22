import torch
import torchtext as text


class Description_embedder:
    def __init__(self, embedding='glove', dimension=50, reduction='mean', authorize_cache=True):
        self.embedder = None
        self.authorize_cache = authorize_cache
        self.cached_embedded_description = dict()
        self.tokenizer = text.data.utils.get_tokenizer(tokenizer="spacy", language="en")
        self.lower_case = True
        if embedding == 'glove':
            self.vocab = text.vocab.GloVe(name='6B', dim=str(dimension))
        else:
            raise NotImplementedError

        if reduction == 'mean':
            self.reduction_func = torch.mean
        elif reduction == "no_reduction":
            self.reduction_func = lambda x: x
        else:
            raise NotImplementedError

    def embed_single_description(self, description):
        assert isinstance(description, str), 'description should be string'
        tokens = self.tokenizer(description)
        description_embedding = self.vocab.get_vecs_by_tokens(tokens, lower_case_backup=self.lower_case)
        description_embedding = self.reduction_func(description_embedding)
        if self.authorize_cache:
            self.cached_embedded_description.update({description: description_embedding})
        return description_embedding

    def embed_description(self, descriptions):
        if isinstance(descriptions, str):
            return self.embed_single_description(descriptions)
        elif isinstance(descriptions, list):
            embedding_lists = []
            for d in descriptions:
                embedding_lists.append(self.embed_single_description(d))
            return embedding_lists
        else:
            raise TypeError("Wrong description format: list or string")

    def get_description_embedding(self, description_list, use_cache=True):
        assert isinstance(description_list, list), 'descriptions should be provided as list'
        embedding_list = []
        for d in description_list:
            assert isinstance(d, str), 'description should be string'
            if use_cache:
                embedded_description = self.cached_embedded_description.get(d, self.embed_single_description(d))
            else:
                embedded_description = self.embed_single_description(d)
            embedding_list.append(embedded_description)
        return embedding_list



