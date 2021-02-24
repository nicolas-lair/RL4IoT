from simulator.Thing import Thing
from simulator.standard_channels import RollerShutterChannel, build_description_and_change_dicts

description_keys = ['roller']
goals = build_description_and_change_dicts(description_keys)


class SimpleBlinds(Thing):
    def __init__(self, name="blinds", description='simple blinds', init_type='random',
                 init_params=None, is_visible=True, location=None, simple=False):
        if simple:
            description = name
        self.roller = RollerShutterChannel()

        super().__init__(name=name, description=description, init_type=init_type,
                         init_params=init_params, is_visible=is_visible, location=location, goals_dict=goals)

    def is_powered(self, state=None):
        return True


if __name__ == "__main__":
    import yaml
    from simulator.oracle import Oracle
    from test_utils import test_action_effect


    def test_goals():
        simple_tv = SimpleBlinds(name='blinds', simple=True)

        oracle = Oracle([simple_tv])
        print('Instruction \n', yaml.dump(oracle.str_instructions))


    ########### Simple Speaker  ###############
    def test_simple_speaker():
        thing = SimpleBlinds(name="blinds", simple=True)
        print('*' * 10 + thing.name + '*' * 10)

        test_action_effect(thing, test_name="Open",
                           init_params=dict(roller=100),
                           action=('roller', 'up')
                           )

        test_action_effect(thing, test_name="Close",
                           init_params=dict(roller=0),
                           action=('roller', 'down')
                           )

        test_action_effect(thing, test_name="Open while open",
                           init_params=dict(roller=0),
                           action=('roller', 'up')
                           )

        test_action_effect(thing, test_name="Close while close",
                           init_params=dict(roller=100),
                           action=('roller', 'down')
                           )


    test_goals()
    test_simple_speaker()
