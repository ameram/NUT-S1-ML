import gym
import numpy as np

class CartPoleQLearning:
    def __init__(self, n_bins=10, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = gym.make('CartPole-v1')
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Define state space bounds
        self.state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.state_bounds[1] = [-4.0, 4.0]  # Cart velocity
        self.state_bounds[3] = [-4.0, 4.0]  # Pole angular velocity
        
        # Initialize Q-table
        self.q_table = np.zeros(([self.n_bins] * len(self.state_bounds) + [self.env.action_space.n]))
    
    def discretize_state(self, state):
        """Convert continuous state into discrete state"""
        discrete_state = []
        for i, (state_val, bounds) in enumerate(zip(state, self.state_bounds)):
            scaled_state = (state_val - bounds[0]) / (bounds[1] - bounds[0])
            discrete_val = min(self.n_bins - 1, max(0, int(scaled_state * self.n_bins)))
            discrete_state.append(discrete_val)
        return tuple(discrete_state)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state + (action,)]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state + (action,)] = new_q
    
    def train(self, n_episodes=1000):
        """Train the Q-learning agent"""
        scores = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):  # Handle new gym API
                state = state[0]
            state = self.discretize_state(state)
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)
                episode_reward += reward
                
                self.update_q_value(state, action, reward, next_state)
                state = next_state
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            scores.append(episode_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return scores
    
    def evaluate(self, n_episodes=100):
        """Evaluate the trained agent"""
        scores = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state = self.discretize_state(state)
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = np.argmax(self.q_table[state])
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)
                episode_reward += reward
                state = next_state
            
            scores.append(episode_reward)
        
        return np.mean(scores)
