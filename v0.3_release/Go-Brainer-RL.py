# tag::load_opponents[]
from dlgo.agent.pg import PolicyAgent
from dlgo.agent.predict import load_prediction_agent
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.rl.simulate import experience_simulation
import h5py

encoder = SevenPlaneEncoder((19,19))

sl_agent = load_prediction_agent(h5py.File('./agents/deep_bot_A0.h5'))
sl_opponent = load_prediction_agent(h5py.File('./agents/deep_bot_A0.h5'))

alphago_rl_agent = PolicyAgent(sl_agent.model, encoder)
opponent = PolicyAgent(sl_opponent.model, encoder)
# end::load_opponents[]

# tag::run_simulation[]
experience = []
for i in range(5):
    num_games = 100
    experience.append(experience_simulation(num_games, alphago_rl_agent, opponent))

for e in experience:
    alphago_rl_agent.train(e)

with h5py.File('alphago_rl_policy.h5', 'w') as rl_agent_out:
    alphago_rl_agent.serialize(rl_agent_out)

with h5py.File('alphago_rl_experience.h5', 'w') as exp_out:
    experience.serialize(exp_out)