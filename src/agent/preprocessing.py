def preprocess_observation(obs):
    for thing_name, thing in obs.items():
        for channel_name, channel in thing.items():
            preprocess_channel(channel)


def preprocess_channel(channel_observation):
    pass
