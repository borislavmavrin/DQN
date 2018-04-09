import os
import gym
import numpy as np
import logging
import time
import sys
from gym import wrappers
from collections import deque

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from utils.test_env import EnvTest

import tensorflow as tf

from configs.q3_nature import config


class DQN(object):
    """
    Abstract Class for implementing a Q Network
    """

    def __init__(self, env, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env

        # build model
        self.build()

    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        self.q = self.get_q_values_op(s, scope="q", reuse=False)

        # compute Q values of next state
        sp = self.process_state(self.sp)
        self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

        # add update operator for target network
        self.update_target_op = self.add_update_target_op("q", "target_q")

        # add square loss
        self.loss = self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.train_op, self.grad_norm = self.add_optimizer_op("q")

    @property
    def policy(self):
        """
        model.policy(state) = action
        """
        return lambda state: self.get_action(state)

    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values

    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]

    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -21.
        self.max_reward = -21.
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.eval_reward = -21.

    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0  # time control of nb of steps
        scores_eval = []  # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            # target_q_norm = self.sess.run(self.target_q_norm)
            # q_norm = self.sess.run(self.q_norm)

            # print(' -- target_q norm: ' + str(target_q_norm))
            # print(' -- q norm: ' + str(q_norm))
            # if target_q_norm == q_norm:
            #     print('-' * 20)
            #     print("q net updated")
            #     print('episode: ' + str(t))
            #     print('-' * 20)
            #     raw_input()

            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train:
                    self.env.render()
                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(
                    t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                        (t % self.config.learning_freq == 0)):
                    self.update_averages(
                        rewards, max_q_values, q_values, scores_eval)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward),
                                                  ("Max R", np.max(
                                                      rewards)), ("eps", exp_schedule.epsilon),
                                                  ("Grads", grad_eval), ("Max Q",
                                                                         self.max_q),
                                                  ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t,
                                                                               self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()

        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval

    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.config.state_history)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test:
                    env.render()

                # store last state in buffer
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, info = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward

    def record(self):
        """
        Re create an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(
            env, self.config.record_path, video_callable=lambda x: True, resume=True)
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                            overwrite_render=self.config.overwrite_render)
        self.evaluate(env, 1)

    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        if self.config.record:
            self.record()

        # model
        self.train(exp_schedule, lr_schedule)

        # record one game at the end
        if self.config.record:
            self.record()

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (80, 80, 4).
               - self.s: batch of states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################
        img_height, img_width, nchannels = state_shape
        self.s = tf.placeholder(
            tf.uint8,
            shape=(None, img_height, img_width,
                   nchannels * self.config.state_history)
        )
        self.a = tf.placeholder(tf.int32, shape=None)
        self.r = tf.placeholder(tf.float32, shape=None)
        self.sp = tf.placeholder(
            tf.uint8,
            shape=(None, img_height, img_width,
                   nchannels * self.config.state_history)
        )
        self.done_mask = tf.placeholder(tf.float32, shape=None)
        self.lr = tf.placeholder(tf.float32, shape=())
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful

        # NOTE:
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param
              make sure to flatten() the tensor before connecting it to fully connected layers 

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################
        # pad to the size used in the paper: 84 x 84
        with tf.variable_scope(scope, reuse=reuse):
            # batch_padded = tf.pad(state, [[0, 0], [2, 2], [2, 2], [0, 0]])
            layer1 = tf.contrib.layers.conv2d(
                inputs=state,
                num_outputs=32,
                kernel_size=[8, 8],
                stride=4,
                padding="SAME",
                activation_fn=tf.nn.relu
            )
            layer2 = tf.contrib.layers.conv2d(
                inputs=layer1,
                num_outputs=64,
                kernel_size=[4, 4],
                stride=2,
                padding="SAME",
                activation_fn=tf.nn.relu
            )

            layer3 = tf.contrib.layers.conv2d(
                inputs=layer2,
                num_outputs=64,
                kernel_size=[3, 3],
                stride=1,
                padding="SAME",
                activation_fn=tf.nn.relu
            )
            layer4 = tf.contrib.layers.fully_connected(
                tf.reshape(layer3, [-1, 10 * 10 * 64]),
                512,
                activation_fn=tf.nn.relu
            )
            layer5 = tf.contrib.layers.fully_connected(
                layer4,
                num_actions,
                activation_fn=None
            )
            out = layer5

        ##############################################################
        ######################## END YOUR CODE #######################
        return out
        ######################## END YOUR CODE #######################

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will
        assign all variables in the target network scope with the values of
        the corresponding variables of the regular network scope.

        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: add an operator self.update_target_op that assigns variables
            from target_q_scope with the values of the corresponding var 
            in q_scope

        HINT: you may find the following functions useful:
            - tf.get_collection
            - tf.assign
            - tf.group

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        # collect variables and assign ops
        # tensorflow guarantees the order of variables in collections
        # is the same as they were added to collections
        target_q_var_lst = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, target_q_scope)

        # print('-' * 20)
        # print(' -- target q')
        # print(len(target_q_var_lst))
        # for var in target_q_var_lst:
        # print(' -- ' + var.name)
        # print('-' * 20)

        q_var_lst = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, q_scope)

        # print('-' * 20)
        # print(' -- q')
        # print(len(q_var_lst))

        # for var in q_var_lst:
        # print(' -- ' + var.name)
        # print('-' * 20)

        assert len(target_q_var_lst) == len(target_q_var_lst), \
            "q and target_q variable list len mismatch"

        update_op_lst = []
        for idx, target_q_var in enumerate(target_q_var_lst):
            op = target_q_var.assign(q_var_lst[idx])
            update_op_lst.append(op.op)
        update_op_grouped = tf.group(*update_op_lst)

        # self.update_target_op = update_op_grouped
        return update_op_grouped

        ##############################################################
        ######################## END YOUR CODE #######################

    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2

              You need to compute the average of the loss over the minibatch
              and store the resulting scalar into self.loss

        HINT: - config variables are accessible through self.config
              - you can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
              - target_q is the q-value evaluated at the s' states (the next states)  
              - you may find the following functions useful
                    - tf.cast
                    - tf.reduce_max / reduce_sum
                    - tf.one_hot
                    - ...

        (be sure that you set self.loss)
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32

        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############
        # batch_size = self.config.batch_size

        # rows = tf.range(0, batch_size)
        # a_idx = tf.stack([rows, tf.cast(self.a, dtype=tf.int32)], axis=1)
        # ones = tf.ones([batch_size])

        # loss_agg = (self.r + self.config.gamma * tf.reduce_max(target_q, axis=1) \
        #           - (ones - self.done_mask) * tf.gather_nd(q, a_idx)) ** 2
        a_one_hot = tf.one_hot(self.a, depth=num_actions)
        loss_agg = self.r + \
            (1. - self.done_mask) * self.config.gamma * tf.reduce_max(target_q, axis=1) - \
            tf.reduce_sum(tf.multiply(q, a_one_hot), axis=1)
        loss = tf.reduce_mean(loss_agg ** 2)
        # self.loss = loss
        return loss
        ##############################################################
        ######################## END YOUR CODE #######################

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        """
        TODO: 1. get Adam Optimizer (remember that we defined self.lr in the placeholders
                section)
              2. compute grads wrt to variables in scope for self.loss
              3. clip the grads by norm with self.config.clip_val if self.config.grad_clip
                is True
              4. apply the gradients and store the train op in self.train_op
               (sess.run(train_op) must update the variables)
              5. compute the global norm of the gradients and store this scalar
                in self.grad_norm

        HINT: you may find the following functinos useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variable by writing self.config.variable_name

        (be sure that you set self.train_op and self.grad_norm)
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############
        var_lst = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        # print('-' * 20)
        # for var in var_lst:
        # print(' -- ' + var.name)
        # print('-' * 20)

        optimizer = tf.train.AdamOptimizer(self.lr)
        # self.train_op = optimizer.minimize(self.loss, var_list=var_lst)
        grads_and_vars_lst = optimizer.compute_gradients(
            self.loss, var_list=var_lst)
        if self.config.grad_clip:
            grads_clipped_and_vars_lst = []
            for grad_and_var in grads_and_vars_lst:
                grad, var = grad_and_var
                grad_clipped = tf.clip_by_norm(grad, self.config.clip_val)
                grads_clipped_and_vars_lst.append((grad_clipped, var))
            # self.train_op = optimizer.apply_gradients(
            #     grads_clipped_and_vars_lst)
            train_op = optimizer.apply_gradients(
                grads_clipped_and_vars_lst)
            grads_lst = [x[0] for x in grads_clipped_and_vars_lst]
        else:
            train_op = optimizer.apply_gradients(grads_and_vars_lst)
            # self.train_op = optimizer.apply_gradients(grads_and_vars_lst)
            grads_lst = [x[0] for x in grads_and_vars_lst]

        # global norm is just a norm of stacked vectors
        # self.grad_norm = tf.global_norm(grads_lst)
        grad_norm = tf.global_norm(grads_lst)
        # var_lst = tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q')
        # debugging: compare norms of target_q and q as an indeirect
        # check of the weight update
        # self.target_q_norm = tf.global_norm(var_lst)
        # var_lst = tf.get_collection(
        # tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
        # self.q_norm = tf.global_norm(var_lst)
        return train_op, grad_norm
        ##############################################################
        ######################## END YOUR CODE #######################

    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state

    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(
            tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(
            tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(
            tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder = tf.placeholder(
            tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder = tf.placeholder(
            tf.float32, shape=(), name="max_q")
        self.std_q_placeholder = tf.placeholder(
            tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(
            tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads_norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max_Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std_Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg_Q", self.avg_q_placeholder)
        tf.summary.scalar("Max_Q", self.max_q_placeholder)
        tf.summary.scalar("Std_Q", self.std_q_placeholder)

        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path,
                                                 self.sess.graph)

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)

        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch,
            self.done_mask: done_mask_batch,
            self.lr: lr,
            # extra info
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.avg_q_placeholder: self.avg_q,
            self.max_q_placeholder: self.max_q,
            self.std_q_placeholder: self.std_q,
            self.eval_reward_placeholder: self.eval_reward,
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm,
                                                               self.merged, self.train_op], feed_dict=fd)

        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

        return loss_eval, grad_norm_eval

    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)


def test_nets_and_update(env, config):
    model = DQN(env, config)

    # inject test data
    s = tf.ones([1, 80, 80, 4], dtype=tf.float32)
    sp = tf.ones([1, 80, 80, 4], dtype=tf.float32)

    # create q_test and target_q_test
    q_test = model.get_q_values_op(s, scope="q_test", reuse=False)
    target_q_test = model.get_q_values_op(
        sp, scope="target_q_test", reuse=False)

    # create update_op

    q_test_var_lst = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        "q_test")
    target_q_test_var_lst = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        "target_q_test")
    update_target_op = model.add_update_target_op("q_test", "target_q_test")

    assert len(q_test_var_lst) == len(target_q_test_var_lst), \
        "number of variables in q and target_q differ"

    # main logic of the test
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    distance_before_lst = []

    # check difference before the update
    for idx in range(len(target_q_test_var_lst)):
        # skip bias, since they are intialized with 0's
        if 'bias' in q_test_var_lst[idx].name:
            continue
        distance_np = (sess.run(tf.norm(
            q_test_var_lst[idx] -
            target_q_test_var_lst[idx]
        )))
        distance_before_lst.append(distance_np)

    assert np.mean(distance_before_lst) != 0., \
        'q and taget_q initialized with the same weights'

    # perform update
    sess.run(update_target_op)

    # check difference after the update
    distance_after_lst = []
    for idx in range(len(target_q_test_var_lst)):
        # skip bias, since they are intialized with 0's
        if 'bias' in q_test_var_lst[idx].name:
            continue
        distance_np = (sess.run(tf.norm(
            q_test_var_lst[idx] -
            target_q_test_var_lst[idx]
        )))
        distance_after_lst.append(distance_np)
    assert np.mean(distance_after_lst) == 0., \
        'q and taget_q weights are different after the update'

    print("network creation and update test passed")


if __name__ == '__main__':
    env = EnvTest((80, 80, 1))
    test_nets_and_update(env, config)
