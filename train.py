import gym
import gym_gomoku


def main():
    env = gym.make('Gomoku9x9-v0')  # default 'beginner' level opponent policy

    # play a game
    env.reset()
    while True:
        action = env.action_space.sample()  # sample without replacement
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == "__main__":
    main()
