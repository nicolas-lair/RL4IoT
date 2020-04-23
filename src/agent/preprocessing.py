def preprocess_observation(obs, description_embedder):
    for thing_name, thing in obs.items():
        for channel_name, channel in thing.items():
            description_embedding = description_embedder.get_description_embedding(channel['description'])



def preprocess_channel(channel_observation):
    description_embedding
