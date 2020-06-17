# This is inspired from IMAGINE project :
# https://github.com/nicolas-lair/Curious-nlp/blob/clean_code/src/architecture_le2/goal_sampler.py

import numpy as np
import torch


# from src import logger


class Goal:
    def __init__(self, goal_string, goal_embedding=None, language_model=None):
        self.goal_embedding = None
        self.goal_string = goal_string
        if goal_embedding is not None:
            self.goal_embedding = goal_embedding
        elif language_model is not None:
            self.compute_embedding(language_model)
        else:
            raise NotImplementedError

        # Added for compatibility, useless
        self.target_counter = 0
        self.reached_counter = 0

    def compute_embedding(self, language_model):
        self.goal_embedding = language_model(self.goal_string).view(1, -1)
        return self.goal_embedding

    @classmethod
    def create_random_goal(cls, goal_embedding_size):
        return cls('', goal_embedding=torch.rand(goal_embedding_size).view(1, -1))


class TrainGoal(Goal):
    def __init__(self, goal_string, episode_discovery, id, target_counter=0, reached_counter=0, goal_embedding=None,
                 language_model=None):
        super().__init__(goal_string=goal_string, goal_embedding=goal_embedding, language_model=language_model)
        self.iter_discovery = episode_discovery
        self.id = id
        self.target_counter = target_counter
        self.reached_counter = reached_counter


class GoalSampler:
    def __init__(self, language_model, goal_sampling_stategy='random', oracle_strategy='exhaustive_feedback'):
        self.discovered_goals = dict()

        self.oracle_strategy = oracle_strategy
        self.goal_sampling_strategy = goal_sampling_stategy
        self.nb_feedbacks = 0
        self.nb_positive_feedbacks = 0

        self.language_model = language_model

    def _update_discovered_goals(self, goal_string, iter):
        if isinstance(goal_string, str):
            assert len(goal_string) > 0, 'goal string should be a non empty string'
            new_goal = TrainGoal(goal_string=goal_string, episode_discovery=iter, id=len(self.discovered_goals),
                                 language_model=self.language_model)
            self.discovered_goals.update({goal_string: new_goal})
        elif isinstance(goal_string, list):
            for g in goal_string:
                self._update_discovered_goals(g, iter=iter)
        else:
            raise TypeError("goals should be passed as a string or list of string")

    def _find_new_goals(self, goals_str):
        if isinstance(goals_str, str):
            s = {goals_str}
        elif isinstance(goals_str, list):
            s = set(goals_str)
        else:
            raise TypeError("goals should be passed as a string or list of string")
        return list(s.difference(list(self.discovered_goals.keys()) + ['']))

    def update(self, target_goals, reached_goals_str, iter):
        new_goals = self._find_new_goals(reached_goals_str)
        self._update_discovered_goals(new_goals, iter=iter)

        self.nb_feedbacks += len(self.discovered_goals)
        self.nb_positive_feedbacks += len(reached_goals_str)

        for g in reached_goals_str:
            self.discovered_goals[g].reached_counter += 1
        for g in target_goals:
            g.target_counter += 1

    def update_embedding(self):
        for g in self.discovered_goals.values():
            g.compute_embedding(self.language_model)

    def sample_goal(self, strategy='random'):
        if len(self.discovered_goals) == 0:
            target = Goal.create_random_goal(goal_embedding_size=self.language_model.embedding_size)
        else:
            strategy = strategy if strategy is not None else self.goal_sampling_strategy
            if strategy == 'random':
                target = np.random.choice(list(self.discovered_goals.values()), 1).item()
            else:
                raise NotImplementedError
        return target

    def get_failed_goals(self, achieved_goals_str):
        if self.oracle_strategy == 'exhaustive_feedback':
            failed_goals = set(self.discovered_goals.keys()).difference(set(achieved_goals_str))
            failed_goals = [self.discovered_goals[g] for g in failed_goals]
        else:
            raise NotImplementedError

        return failed_goals
