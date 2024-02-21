import jax
import jax.numpy as jnp
from jumanji.environments.logic.game_2048.env import Game2048
from jumanji.wrappers import AutoResetWrapper, VmapWrapper

from ppo2048.policies import Policy
from ppo2048.types import NetworkParams

SEED = 0
SIZE = 4
PLAY_BATCH_SIZE = 2
PLAY_STEPS_PER_EPOCH = 5
key = jax.random.PRNGKey(SEED)
env = VmapWrapper(AutoResetWrapper(Game2048(SIZE)))
key, subkey = jax.random.split(key)
subkeys = jax.random.split(subkey, PLAY_BATCH_SIZE)
init_state, _ = env.reset(jnp.array(subkeys))


def policy_check(key, policy: Policy, params: NetworkParams):
    rollout_fn = jax.jit(policy.rollout, static_argnums=(0, 1))
    state, rollout = rollout_fn(env, PLAY_STEPS_PER_EPOCH, key, init_state, params)
    print(rollout.action)
