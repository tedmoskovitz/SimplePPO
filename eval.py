from flax.training.train_state import TrainState
import gymnax
import jax
import pdb
import pickle
from ppo import ActorCritic, Transition
from utils import load_checkpoint
from wrappers import LogWrapper, FlattenObservationWrapper

######################
# settings
run_name = "ppo"
save_step = 1560
checkpoint = f"checkpoints/{run_name}_{save_step}.flax"
with open(f"configs/{run_name}_config.pkl", "rb") as f:
  config = pickle.load(f)
config["NUM_ENVS"] = 1
# environment setup
print(f"Loading {config['ENV_NAME']} environment.")
env, env_params = gymnax.make(config["ENV_NAME"])
env = FlattenObservationWrapper(env)
env = LogWrapper(env)
# agent setup
print(f"Initializing agent from checkpoint.")
agent = ActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
checkpoint_dict = load_checkpoint(checkpoint, agent)
agent_params = checkpoint_dict["params"]

######################
# Run evaluation

# INIT ENV
rng = jax.random.PRNGKey(30)
rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

# COLLECT TRAJECTORIES
def _env_step(runner_state, unused):
    agent_params, env_state, last_obs, rng = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi, value = agent.apply(agent_params, last_obs)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(rng_step, env_state, action, env_params)
    transition = Transition(
        done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (agent_params, env_state, obsv, rng)
    return runner_state, transition

runner_state = (agent_params, env_state, obsv, _rng)

# returns final runner_state tuple, and traj_batch is a Transition containting
# action, obs, reward, done, log_prob, value, info, each of which is
# (T, NUM_ENVS, ...)
# info is dict_keys(['discount', 'returned_episode', 'returned_episode_lengths', 
#                    'returned_episode_returns', 'timestep'])
runner_state, traj_batch = jax.lax.scan(
    _env_step, runner_state, None, config["NUM_STEPS"]
)
pdb.set_trace()

print("Done.")


"""
def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)
"""

