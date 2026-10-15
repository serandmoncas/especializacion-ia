import numpy as np
from ai_foundations.agents import (
    VacuumWorld,
    model_based_agent,
    random_agent,
    run_episode,
    simple_reflex_agent,
)


def mean_clean(make_agent, seeds=range(5)):
    out = []
    for s in seeds:
        world = VacuumWorld(6, 6, dirt_prob=0.01, rng=np.random.default_rng(s))
        out.append(run_episode(make_agent(), world, steps=600).clean_fraction)
    return float(np.mean(out))


def test_agents_ranking():
    rand = mean_clean(lambda: random_agent(np.random.default_rng(0)))
    reflex = mean_clean(lambda: simple_reflex_agent(np.random.default_rng(0)))
    model = mean_clean(lambda: model_based_agent(6, 6))
    assert rand < reflex < model


def test_world_stays_in_bounds_and_suck_cleans():
    world = VacuumWorld(3, 2, dirt_prob=0.0, rng=np.random.default_rng(0))
    for action in ["left"] * 5 + ["up"] * 5:
        world.step(action)
    assert world.pos == (0, 0)
    world.dirty[0, 0] = True
    world.step("suck")
    assert not world.dirty[0, 0]
