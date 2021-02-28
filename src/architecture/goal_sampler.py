import numpy as np
import torch
import joblib

from logger import get_logger

logger = get_logger(__name__)


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

    # For compatibility issue
    def update_record(self, **kwargs):
        pass

    def __eq__(self, other):
        return self.goal_string == other.goal_string


class TrainGoal(Goal):
    def __init__(self, goal_string, episode_discovery, target_counter=0, reached_counter=0, goal_embedding=None,
                 language_model=None):
        super().__init__(goal_string=goal_string, goal_embedding=goal_embedding, language_model=language_model)
        self.record = dict(
            iter_discovery=episode_discovery,
            target_counter=target_counter,
            reached_counter=reached_counter
        )
        # self.iter_discovery = episode_discovery
        # # self.id = id
        # self.target_counter = target_counter
        # self.reached_counter = reached_counter

    def update_record(self, targeted=False, reached=False):
        if targeted:
            self.record['target_counter'] += 1
        if reached:
            self.record['reached_counter'] += 1


class GoalSampler:
    def __init__(self, language_model, goal_sampling_stategy='random', oracle_strategy='exhaustive_feedback'):
        self.discovered_goals = dict()

        self.oracle_strategy = oracle_strategy
        self.goal_sampling_strategy = goal_sampling_stategy

        self.record = dict(nb_feedbacks=0,
                           nb_positive_feedbacks=0,
                           target_goal_sequence=[],
                           reached_goal_sequence=dict())
        self.language_model = language_model

    def update_discovered_goals(self, goals_str, iter):
        if isinstance(goals_str, str):
            s = {goals_str}
        elif isinstance(goals_str, list):
            s = set(goals_str)
        else:
            raise TypeError("goals should be passed as a string or list of string")
        new_goals_str = list(s.difference(list(self.discovered_goals.keys()) + ['']))
        if new_goals_str:
            logger.info(f'New goals discovered: {new_goals_str}')
        for g in new_goals_str:
            assert len(g) > 0, 'goal string should be a non empty string'
            new_goal = TrainGoal(goal_string=g, episode_discovery=iter, language_model=self.language_model)
            self.discovered_goals.update({g: new_goal})

    def update(self, reached_goals_str, iter):
        logger.debug('Updating Goal Sampler')
        self.update_discovered_goals(reached_goals_str, iter=iter)
        self.update_records(reached_goals_str, iter)

    def update_records(self, reached_goals_str, iter):
        self.record['nb_feedbacks'] += len(self.discovered_goals)
        self.record['nb_positive_feedbacks'] += len(reached_goals_str)
        self.record['reached_goal_sequence'][iter] = reached_goals_str

        for g in reached_goals_str:
            self.discovered_goals[g].update_record(reached=True)

    def update_embedding(self):
        for g in self.discovered_goals.values():
            g.compute_embedding(self.language_model)

    def sample_goal(self, strategy='random'):
        if len(self.discovered_goals) == 0:
            target = Goal.create_random_goal(goal_embedding_size=self.language_model.out_features)
        else:
            strategy = strategy if strategy is not None else self.goal_sampling_strategy
            if strategy == 'random':
                target = np.random.choice(list(self.discovered_goals.values()), 1).item()
            else:
                raise NotImplementedError
            target.update_record(targeted=True)
            self.record['target_goal_sequence'].append(target.goal_string)

        return target

    def get_failed_goals(self, achieved_goals_str):
        if self.oracle_strategy == 'exhaustive_feedback':
            failed_goals = set(self.discovered_goals.keys()).difference(set(achieved_goals_str))
            failed_goals = [self.discovered_goals[g] for g in failed_goals]
        else:
            raise NotImplementedError

        return failed_goals

    def get_records(self):
        record = dict(goal_sampler_record=self.record,
                      goals_record={v.goal_string: v.record for v in self.discovered_goals.values()}
                      )
        return record

    def get_record_for_logging(self):
        d ={
            g.goal_string: g.record for g in self.discovered_goals.values()
        }
        d['nb_feedbacks'] = self.record['nb_feedbacks'],
        d['nb_positive_feedbacks'] = self.record['nb_positive_feedbacks'],
        # d['overall'] = self.record
        return d
