class Opponent(object):
    def __init__(self, flatten_obs, replay_buffer, act):
        self.old_obs = None
        self.old_action = None
        self.__obs = None
        self.__flatten_obs = flatten_obs
        self.__replay_buffer = replay_buffer
        self.__act = act

    def policy(self, curr_state, prev_state, prev_action):
        '''
        Define policy for opponent here
        '''
        self.__obs = curr_state.encode()

        if self.old_obs is not None:
            self.__replay_buffer.add(self.old_obs, self.old_action,
                                     0, self.__obs, 0)
        # Get opponent action
        # if self.__flatten_obs:
        #     self.__obs = self.__obs.flatten()
        action = self.__act(self.__obs[None])[0]

        self.old_obs = self.__obs
        self.old_action = action
        return action
