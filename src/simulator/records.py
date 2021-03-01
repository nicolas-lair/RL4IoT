from collections import deque
import joblib
import yaml
from pathlib import Path

import numpy as np

from logger import get_logger

logger = get_logger(__name__)


class RollingAverage(deque):
    def ___init__(self, maxlen=5):
        super().__init__(maxlen=maxlen)

    def get_mean(self):
        return np.round(np.mean(self), 2)


class Records:
    def __init__(self, save_path=None, rolling_window=5):
        self.test_record = dict()
        self.goal_sampler_records = dict()

        self.best_result = (-1, {'overall': 0})
        self.train_rolling_average = RollingAverage(maxlen=rolling_window)
        self.new_objet_introduction_episode = -1

        self.save_path = Path(save_path)
        self.model_path = self.save_path.joinpath('agent')
        self.model_path.mkdir()

    def update_test_records(self, episode, test_scores, train_goals=None):
        d = dict()
        for thing_name, thing_score in test_scores.items():
            d[thing_name] = thing_score.copy()

            overall_thing = float(np.round(np.mean(list(thing_score.values())), 2))
            d[thing_name][f'overall {thing_name}'] = overall_thing

        thing_list = list(test_scores)
        if train_goals is not None:
            d['overall train'] = self.compute_average_scores_over_list_of_goals(test_scores, keep_list=train_goals)
            self.update_rolling_average(d['overall train'])
            d['overall test'] = self.compute_average_scores_over_list_of_goals(test_scores, exclude_list=train_goals)
        d['overall'] = self.compute_average_scores_over_list_of_goals(test_scores)
        self.test_record[episode] = d

        logger.info("%" * 5 + f" Test after {episode} episodes " + "%" * 5 + "\n" + yaml.dump(d, sort_keys=False))

    def keep_best_agent(self, episode, agent):
        if self.test_record[episode]['overall'] > self.best_result[1]['overall']:
            self.best_result = (episode, self.test_record[episode])
            self.save_best_agent(agent)
            logger.info('This is best result!')
        else:
            logger.info("%" * 5 + f" Best result after {self.best_result[0]} episodes " + "%" * 5 + "\n" + yaml.dump(
                self.best_result[1], sort_keys=False))

    def update_rolling_average(self, score):
        self.train_rolling_average.append(score)
        logger.info(
            f" Train rolling average over {self.train_rolling_average.maxlen} tests: {self.train_rolling_average.get_mean()}")

    def update_goal_sampler_records(self, goal_sampler):
        self.goal_sampler_records = goal_sampler.get_records()
        # logger.info(f"Goal sampler record: \n {yaml.dump(agent.goal_sampler.get_record_for_logging())}")

    def save_best_agent(self, agent):
        agent.save(self.model_path)

    def update_records(self, agent, episode, test_scores, train_goals=None):
        self.update_test_records(episode, test_scores, train_goals)
        self.keep_best_agent(episode, agent)
        self.update_goal_sampler_records(agent.goal_sampler)

    @staticmethod
    def compute_average_scores_over_list_of_goals(test_scores, keep_list=None, exclude_list=None):
        """
        Used for computing score over train
        Parameters
        ----------
        test_scores :
        keep_list :
        exclude_list :

        Returns
        -------

        """
        d = dict()
        for thing_name, thing_score in test_scores.items():
            d.update(thing_score)

        if keep_list is None and exclude_list is None:
            d = list(d.values())
        elif keep_list is not None and exclude_list is None:
            d = [d[k] for k in keep_list]
        elif keep_list is not None and exclude_list is None:
            d = [d[k] for k in set(d.keys()).difference(exclude_list)]
        else:
            raise EOFError('keep_list and exclude_list cannot be both None')

        return float(np.round(np.mean(d), 2))

    def save(self):
        # Test record
        joblib.dump(self.test_record, self.save_path.joinpath('test_record.jbl'))

        # Goal sampler
        joblib.dump(self.goal_sampler_records, self.save_path.joinpath('goal_sampler_record.jbl'))

        # Best result
        joblib.dump(self.best_result, self.save_path.joinpath('best_result.jbl'))

    def save_as_object(self):
        joblib.dump(self, self.save_path.joinpath('records_object.jbl'))
