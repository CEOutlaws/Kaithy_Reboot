import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import os
from math import floor, ceil, pi




def rotate_images(X_imgs):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate
def rotate(X_imgs):
    rotated_imgs = rotate_images(X_imgs)
    print(rotated_imgs.shape)


        # obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        # act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        # rew_t_ph = tf.placeholder(U.data_type, [None], name="reward")
        # obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))

def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)

def main():
    '''
    AI Self-training program
    '''
    deterministic_filter = True
    random_filter = True
    env = gym.make('Gomoku5x5-training-camp-v0', opponent_policy)

    num_actions = env.action_space.n

    # obs_ph = tf.placeholder(
    #     dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
    # q_values = layers.fully_connected(layers.flatten(obs_ph), num_actions)
    def make_obs_ph(name):
        obs_shape = env.observation_space.shape

        if flatten_obs:
            flattened_env_shape = 1
            for dim_size in env.observation_space.shape:
                flattened_env_shape *= dim_size
            obs_shape = (flattened_env_shape,)

        return U.BatchInput(obs_shape, name=name)
    # obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
    # obs_t_input = tf.placeholder(
    #     dtype=tf.float32,shape=[None])
    size =32
    obs_t_input = tf.placeholder(
        dtype=tf.float32, shape=list(env.observation_space.shape))

    act_t_ph = tf.placeholder(tf.int32, [None], name="action")
    obs_tp1_input = tf.placeholder(
        dtype=tf.float32, shape=[None] + list(env.observation_space.shape))

    

    # batch_size = tf.shape(obs_t_input)[0]
    # print(list(env.observation_space.shape),tf.shape(obs_t_input[1:4]))
    # exit(0)
    list_obs = []
    # for i in range(0,8): 
    #     if (i > 0 and i <4):
    #         obs_t_input[i,:, :, :] = tf.image.rot90(obs_t_input[1,:,:,:], k = i)
    #     if (i ==4 ) :
    #         obs_t_input[i] = tf.image.flip_left_right(obs_temp_ph[1:4])
    #     if (i>4 and i <8):
    #         obs_t_input[i] = tf.image.rot90(obs_t_input[4], k = (i-4))
    list_obs.append(obs_t_input)
    for i in range(0,8): 
        if (i > 0 and i <4):
            list_obs.append(tf.image.rot90(obs_t_input, k=i))
        if (i ==4 ) :
            list_obs.append(tf.image.flip_left_right(obs_t_input))
        if (i>4 and i <8):
            list_obs.append(tf.image.rot90(obs_t_input, k = (i-4)))


    obs_ph = tf.stack(list_obs)
    # print(tf.shape(obs_ph)[0])
    # exit(0)
    # obs_ph = tf.placeholder(
    #     dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
    # print(tf.shape(obs_ph)[0])
    # exit(0)
    q_values = layers.fully_connected(layers.flatten(obs_ph), num_actions)
    if deterministic_filter or random_filter:
        invalid_masks = tf.contrib.layers.flatten(
            tf.reduce_sum(obs_ph[:, :, :, 1:3], axis=3))
        # print(tf.shape(invalid_masks))
        # exit(0)

    if deterministic_filter:
        q_values_worst = tf.reduce_min(q_values, axis=1, keep_dims=True)
        # q_values = tf.where(tf.equal(
        #     invalid_masks, 1.), q_values_worst - 1.0, q_values)
        q_values = invalid_masks * (q_values_worst - 1.0) + \
            (1.0 - invalid_masks) * q_values

    deterministic_actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
    batch_size = tf.shape(obs_ph)[0]
    stochastic_ph = tf.constant(True, dtype=tf.bool)
    random_actions = tf.random_uniform(
        tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int32)

    if random_filter:
        def get_elements(data, indices):
            indeces = tf.range(0, tf.shape(indices)[
                0]) * data.shape[1] + indices
            return tf.gather(tf.reshape(data, [-1]), indeces)
        is_invalid_random_actions = get_elements(
            invalid_masks, random_actions)
        random_actions = tf.where(tf.equal(
            is_invalid_random_actions, 1.), deterministic_actions, random_actions)

    chose_random = tf.random_uniform(
        tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < 0.9
    stochastic_actions = tf.where(
        chose_random, random_actions, deterministic_actions)

    output_actions = tf.where(
        stochastic_ph, stochastic_actions, deterministic_actions)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    alo = optimizer.minimize(q_values)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    observations = []

    for i in range(2):
        observation = env.reset()
        # print(observation)
        done = None
        
        # print(observation.shape)
        # exit(0)
        while not done:
            def rotate_action(board_size,pos_1D,k):
                pos_2D = (pos_1D // board_size , pos_1D % board_size)
                # rot90
                if (k==1):
                    rot_pos = pos_2D[0]*board_size+(board_size-1 - pos_2D[1] ) 
                # rot180
                if (k==2):
                    rot_pos = (board_size -1 - pos_2D[0] )*board_size + (board_size-1 -pos_2D[1])
                # rot270
                if (k==3):
                    rot_pos = (board_size-1-pos_2D[0]) + pos_2D[1]*board_size
                return rot_pos
            def flip_action(board_size,pos_1D,k):
                pos_2D = (pos_1D // board_size , pos_1D % board_size)
                # flip and rot 0
                if (k==0):
                    flip_rot = pos_2D[0]*board_size + -pos_2D[1]+board_size-1
                # flip and rot 90
                if (k==1):
                    flip_rot = pos_2D[1]*board_size+pos_2D[0]
                # flip and rot 180
                if (k==2):
                    flip_rot = (-pos_2D[0]+board_size -1)*board_size +pos_2D[1]
                # flip and rot 270
                if (k==3):
                    flip_rot = (-pos_2D[1]+board_size -1 )*board_size+ -pos_2D[0]+board_size-1
                return flip_rot

            actions = sess.run(output_actions, feed_dict={
                obs_t_input: observation})
            action = actions[0]
            for i in range(1,8):
                if (i<4):
                    actions[i] = rotate_action(observation.shape[0],action,i)
                else :
                    actions[i] = flip_action(observation.shape[0],action,(i-4))
            print(actions,observation[:,:,1])
            exit(0)
            observation, reward, done, info = env.step(action)
            
            #  start action rotate

            def rotate_action(board_size,pos_1D,k):
                pos_2D = (pos_1D // board_size , pos_1D % board_size)
                # rot90
                if (k==1):
                    rot_pos = pos_2D[0]*board_size+(board_size-1 - pos_2D[1] ) 
                # rot180
                if (k==2):
                    rot_pos = (board_size -1 - pos_2D[0] )*board_size + (board_size-1 -pos_2D[1])
                # rot270
                if (k==3):
                    rot_pos = (board_size-1-pos_2D[0]) + pos_2D[1]*board_size
                print(rot_pos,pos_1D,pos_2D)
                # exit(0)
            def flip_action(board_size,pos_1D,k):
                pos_2D = (pos_1D // board_size , pos_1D % board_size)
                # flip and rot 0
                if (k==0):
                    flip_rot = (pos_2D[0],-pos_2D[1]+board_size-1)
                # flip and rot 90
                if (k==1):
                    flip_rot = (pos_2D[1],pos_2D[0])
                # flip and rot 180
                if (k==2):
                    flip_rot = (-pos_2D[0]+board_size -1 ,pos_2D[1])
                # flip and rot 270
                if (k==3):
                    flip_rot = (-pos_2D[1]+board_size -1 , -pos_2D[0]+board_size-1)


                print(flip_rot,pos_1D,pos_2D)
            # print(action)
            angle = 0
            flip_action(observation.shape[0],action,angle)
            # exit(0)
            #  observation flip and rotate
            print(observation[:, :, 1],env.observation_space.shape[0:2])
            # exit(0)
            obs_temp_ph = tf.placeholder(dtype=tf.int32, shape=(env.observation_space.shape))
            k = tf.placeholder(tf.int32)
            # tf_img = tf.image.rot90(obs_temp_ph, k = k)
            # tf_img1 = tf.image.flip_left_right(obs_temp_ph[0:3])
            tf_img = tf.image.rot90(obs_temp_ph, k = k)
            # tf_img = tf.image.flip_left_right(obs_temp_ph)
            rotated_img = sess.run(tf_img, feed_dict = {obs_temp_ph: observation, k:angle})
            # rotated_img = sess.run(tf_img1, feed_dict = {obs_temp_ph: observation})
            print(rotated_img[:, :, 1])
            exit(0)
            
            # end obser
            env.render()
            observations.append(observation)

        print(reward)
        env.swap_role()
        print("\n----SWAP----\n")

    # actions = sess.run(output_actions, feed_dict={
    #     obs_ph: observations})
    # sess.run(q_values, feed_dict={
    #     obs_ph: observations})
    # print(actions)

if __name__ == "__main__":
    main()