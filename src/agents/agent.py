from src import Trainable
from src.models.model import Model
from src.env_builders import EnvBuilder
from abc import abstractmethod
from torch.types import Tensor
from torch.distributions import Categorical

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.configs.agent import AgentConfig

class Agent(Trainable):
    """Abstract base class for all actor-critic agents."""

    def __init__(self, cfg: 'AgentConfig', state: dict = None) -> None:
        super().__init__(cfg, state)
        self.cfg = cfg

        # Builds the environments from the env config
        self.envs = EnvBuilder.build(cfg.environment)
        self.output_size = self.envs.single_action_space.n

        # Builds the actor and critic models
        self.actor = Model.from_name(cfg.type + cfg.actor, output_size=self.output_size)
        self.critic = Model.from_name(cfg.type + cfg.critic)

        # Loads state if it was provided
        if state is not None:
            self.actor.load_state_dict(state['actor'])
            self.critic.load_state_dict(state['critic'])

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict['actor'] = self.actor.state_dict()
        state_dict['critic'] = self.critic.state_dict()
        return state_dict

    def get_action(self, observation: Tensor) -> Tensor:
        """Returns a sampled action from the actor given an observation."""
        return Categorical(logits=self.actor(observation)).sample()
    
    def get_action_logits(self, observation: Tensor) -> Tensor:
        """Returns the logits for all actions from the actor given an observation."""
        return self.actor(observation)
    
    def get_value(self, observation: Tensor) -> Tensor:
        """Returns the state value calculated by the critic for the given observation."""
        return self.critic(observation)
    
    def get_action_and_value(
        self, observation: Tensor, action: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Bundles actor and critic evaluation to return a sampled action and value.

        Parameters
        ----------
        observation : Tensor
            Current observation from the environment.
        action : Tensor | None, optional
            Specified action to be taken in the current state, by default None,
            in which case the action is sampled from the actor network for rollout

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            - Sampled action to be taken given the observation (or the specified value)
            - Log probability of the returned action
            - Entropies of the action probability distribution
            - Critic value of the observation
        """
        # Gets un-normalised action probabilites from the actor network
        logits = self.actor(observation)

        # Uses a categorical distribution over the actor's output for the actions
        probs = Categorical(logits=logits)

        # If no action was specified, sample it from the probabilities
        action = action if action is not None else probs.sample()

        # Returns the tuple of action details and value
        return action, probs.log_prob(action), probs.entropy(), self.critic(observation)
