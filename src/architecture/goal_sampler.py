# This is inspired from IMAGINE project :
# https://github.com/nicolas-lair/Curious-nlp/blob/clean_code/src/architecture_le2/goal_sampler.py

import numpy as np


# from src import logger


class Goal:
    def __init__(self, goal_string, iter_discovery, id, target_counter=0, reached_counter=0, goal_embedding=None,
                 language_model=None):
        self.iter_discovery = iter_discovery
        self.id = id
        self.goal_embedding = None
        self.goal_string = goal_string
        self.target_counter = target_counter
        self.reached_counter = reached_counter
        if language_model is not None:
            self.update_embedding(language_model)
        if goal_embedding is not None:
            self.goal_embedding = goal_embedding

    def update_embedding(self, language_model):
        self.goal_embedding = language_model(self.goal_string)


class GoalSampler:
    def __init__(self, language_model):
        self.discovered_goals = dict()

        self.nb_feedbacks = 0
        self.nb_positive_feedbacks = 0

        self.language_model = language_model

    def _update_discovered_goals(self, goal_string, iter):
        if isinstance(goal_string, str):
            new_goal = Goal(goal_string=goal_string, iter_discovery=iter, id=len(self.discovered_goals),
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
        return list(s.difference(self.discovered_goals.keys()))

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
            g.update_embedding(self.language_model)

    def sample_goal(self, strategy='random'):
        if strategy == 'random':
            target = np.random.choice(list(self.discovered_goals.values()), 1).item()
        else:
            raise NotImplementedError
        return target
