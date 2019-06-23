class Environment:
    """A wrapper for the environment."""
    def __init__(self, unity_environment):
        """Initialize an environement object.
        
        Params  
        ======
            unity_environment (UnityEnvironement): An instance of the unity environment desired
        """
        self.env = unity_environment
        self.brain_name = self.env.brain_names[0]
        
    def step(self, action):
        """Takes a single step through the environment and returns the next_state, the corresponding reward and if the agent is complete.
        
        Params  
        ======
            actior (int): An action the agent has taken.
        """
        res = self.env.step(int(action))[self.brain_name]
        next_state = res.vector_observations[0]
        reward = res.rewards[0]  
        done = res.local_done[0] 
        return next_state, reward, done
    
    def reset(self, is_train_mode=True):
        """Resets the environment and returns the initial state
        
        Params  
        ======
            is_train_mode (boolean): If true, train_mode applied else false.
        """
        env_info = self.env.reset(train_mode=is_train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def close(self):
        """Closes the environment"""
        self.env.close()
        