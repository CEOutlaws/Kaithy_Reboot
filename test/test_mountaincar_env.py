import gym
from random import randint


def main():
    env = gym.make("MountainCar-v0")

    while True:
        obs, done = env.reset(), False
        a = env.action_space
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(randint(0, 2))
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == "__main__":
    main()
