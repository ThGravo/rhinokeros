from agent import Agent
import model_builders as mb
from policy import DirectPolicy, ValuePolicy
import numpy as np
import heapq


class CEMAgent(Agent):
    """ Cross-Entropy Method for maximizing a black-box function
    """

    def __init__(self, observation_space, action_space,
                 samples_per_iteration=50, steps_per_sample=2017, elite_frac=0.1,
                 initial_mean=0.0, initial_std=10, Z=10, Z_decay_factor=.9, policy_based=True):
        """
        initial_mean: initial mean over input distribution
        initial_std: initial standard deviation over parameter vectors
        K, K_decay_factor: http://iew3.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf
        elite_frac: each batch, select this fraction of the top-performing samples
        """
        super().__init__(observation_space, action_space)

        if policy_based:
            self.model = mb.build_policy_model(observation_space, action_space)
            self.policy = DirectPolicy(action_space, self.model)
        else:
            self.model = mb.build_q_model(observation_space, action_space)
            self.policy = ValuePolicy(action_space, self.model)

        self.elite_frac = elite_frac
        self.elite_num = round(samples_per_iteration * elite_frac)
        self.elite = []
        self.reset_elite_set()

        self.mu = [np.zeros(w.shape) + initial_mean for w in self.model.get_weights()]
        self.std = [np.ones(w.shape) * np.sqrt(np.square(initial_std) + Z) for w in self.model.get_weights()]

        self.steps_per_sample = steps_per_sample
        self.samples_per_iteration = samples_per_iteration
        self.current_steps = 0
        self.current_sample_acc_reward = 0
        self.done_count = 0

        self.Z = Z
        self.Z_decay_factor = Z_decay_factor

    def save_weights(self, filepath, overwrite=True):
        self.update_theta_distribution()
        self.model.set_weights(self.mu)
        self.model.save_weights(filepath, overwrite)

    def reset_elite_set(self):
        self.elite = [(-np.inf, i, self.model.get_weights()) for i in range(self.elite_num)]
        heapq.heapify(self.elite)

    def update_theta_distribution(self):
        self.Z *= self.Z_decay_factor
        self.mu = [np.zeros(w.shape) for w in self.model.get_weights()]
        self.std = [np.ones(w.shape) for w in self.model.get_weights()]
        # sample mean
        for el in self.elite:
            for layer, wghts in enumerate(el[2]):
                self.mu[layer] += wghts
        for layer, wghts in enumerate(self.mu):
            self.mu[layer] /= len(self.elite)
        # sample standard deviation
        for el in self.elite:
            for layer, wghts in enumerate(el[2]):
                self.std[layer] += np.square(wghts - self.mu[layer])
        for layer, wghts in enumerate(self.mu):
            self.std[layer] = np.sqrt(self.std[layer] / (len(self.elite) - 1) + self.Z)

    def draw_theta(self):
        weights = []
        for layer, wghts in enumerate(self.mu):
            weights.append(self.mu[layer] + self.std[layer] * np.random.standard_normal(wghts.shape))
        return weights

    def setup_another_theta_sample(self):
        heapq.heappushpop(self.elite,
                          (self.current_sample_acc_reward / max(self.done_count, 1),
                           self.current_steps / self.steps_per_sample,
                           self.model.get_weights()))
        self.model.set_weights(self.draw_theta())

    def act(self, state, prev_reward, prev_done):
        self.current_steps += 1
        self.current_sample_acc_reward += prev_reward

        if self.current_steps % self.steps_per_sample == 0:
            print('Sample %2i in iteration %2i. Total reward of %7.3f gained in %2i episodes. Mean reward: %7.3f' % (
                self.current_steps / self.steps_per_sample % self.samples_per_iteration,
                self.current_steps / (self.steps_per_sample * self.samples_per_iteration),
                self.current_sample_acc_reward, self.done_count,
                self.current_sample_acc_reward / max(self.done_count, 1)))

            if self.current_steps % (self.steps_per_sample * self.samples_per_iteration) == 0:
                self.update_theta_distribution()
                self.reset_elite_set()

            self.setup_another_theta_sample()
            self.current_sample_acc_reward = 0
            self.done_count = 0

        self.done_count += 1*prev_done

        return self.policy.get_action(state), self.current_steps % self.steps_per_sample == 0
