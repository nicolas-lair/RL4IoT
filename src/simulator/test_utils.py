from simulator.Items import MethodUnavailableError


def test_action_effect(thing, test_name, init_params, action, debug=False):
    print('     ' + '*' * 3 + f'{test_name}' + '*' * 3)
    s1 = thing.init_node(is_visible=True, init_type='custom', init_params=init_params)
    print(f"Current thing descriptions: {[d.get_instruction() for d in thing.get_thing_description()]}")
    action = action if isinstance(action, list) else [action]
    if debug: print(s1)
    for a in action:
        print(f"Current thing descriptions: {[d.get_instruction() for d in thing.get_thing_description()]}")
        try:
            thing.do_action(*a)
            s2 = thing.get_state(oracle=True)
            print(f"{sorted(thing.get_thing_change(s1, s2))} after action {a}")
            if debug: print(s2)
            s1 = s2
        except MethodUnavailableError:
            s2 = thing.get_state(oracle=True)
            print(f'Action {a} is not available for {thing.name}', sorted(thing.get_thing_change(s1, s2)))
            if debug: print(s2)
            s1 = s2
