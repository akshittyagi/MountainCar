import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum
import multiprocessing
from multiprocessing import Pool
import time
import math

import numpy as np
from matplotlib import pyplot as plt
from DecisionProcess import MDP

class TD(object):

    def __init__(self, mdp, num_training_episodes=100, order=3, alpha=0.01):
        self.mdp = mdp
        self.num_train = num_training_episodes
        self.alpha = alpha
        self.gamma = 1
        self.order = order
    
    def initialize_weights(self):
        self.w = np.zeros((self.order + 1) ** self.mdp.states)
    
    def initialize_weights_e_trace(self):
        self.e = np.zeros((self.order + 1) ** self.mdp.states)
        tmp = self.e
        return self.e

class Sarsa(TD):

    def __init__(self, mdp, fourier_order, epsilon, alpha, num_training_episodes):
        super(Sarsa, self).__init__(mdp, alpha=alpha, order=fourier_order)
        self.episodes = num_training_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma
    
    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / 1
        q_a_1, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['forward'])
        q_a_2, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['neutral'])
        q_a_3, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['reverse'])
        q_values = [q_a_1, q_a_2, q_a_3]
        if self.debug:
            print "QFOR: ", q_values[0], " QNEUT: ", q_values[1], "QREV: ", q_values[2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            return argmax[0]
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            return coin_toss
    
    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        
        self.initialize_weights()
        self.debug = debug

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            a_t = self.epsilon_greedy_action_selection(s_t)
            time_step = 0
            temperature = 1
            g = 0
            while not self.mdp.is_terminal_state(s_t) and time_step <= 1000:
                alpha = alpha / temperature
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                a_t_1 = self.epsilon_greedy_action_selection(s_t_1, temperature=temperature)
                q_s_prime_a_prime, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, a_t_1)
                q_s_a, dq_dw = self.mdp.get_q_value_function(self.order, self.w, s_t, a_t)
                q_td_error = ((r_t + self.gamma*(q_s_prime_a_prime) - q_s_a)*dq_dw)
                if self.debug:
                    print "QERROR: ", np.sqrt(np.sum(q_td_error**2))
                    print "S_T: ", s_t, "A_T: ", a_t, "S_T+1: ", s_t_1, "A_T+1: ", a_t_1 
                self.w += alpha*(q_td_error)
                s_t = s_t_1
                a_t = a_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                # temperature = global_time_step**(1.0/reduction_factor)
                # temperature = temperature / (global_time_step + 2)
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)

class SarsaLambda(TD):

    def __init__(self, mdp, fourier_order, epsilon, alpha, num_training_episodes):
        super(SarsaLambda, self).__init__(mdp, alpha=alpha, order=fourier_order)
        self.episodes = num_training_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma
        print "SARSA LAMBDA"
    
    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / 1
        q_a_1, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['forward'])
        q_a_2, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['neutral'])
        q_a_3, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['reverse'])
        q_values = [q_a_1, q_a_2, q_a_3]
        if self.debug:
            print "QFOR: ", q_values[0], " QNEUT: ", q_values[1], "QREV: ", q_values[2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            return argmax[0]
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            return coin_toss
    
    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        
        self.initialize_weights()
        self.debug = debug
        lamb = 0.9

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            a_t = self.epsilon_greedy_action_selection(s_t)
            time_step = 0
            temperature = 1
            g = 0

            self.initialize_weights_e_trace()

            while not self.mdp.is_terminal_state(s_t) and time_step <= 1000:
                alpha = alpha / temperature
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                a_t_1 = self.epsilon_greedy_action_selection(s_t_1, temperature=temperature)
                q_s_prime_a_prime, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, a_t_1)
                q_s_a, dq_dw = self.mdp.get_q_value_function(self.order, self.w, s_t, a_t)
                q_td_error = ((r_t + self.gamma*(q_s_prime_a_prime) - q_s_a))
                self.e = lamb*self.gamma*self.e + dq_dw
                if self.debug:
                    print "QERROR: ", np.sqrt(np.sum(q_td_error**2))
                    print "S_T: ", s_t, "A_T: ", a_t, "S_T+1: ", s_t_1, "A_T+1: ", a_t_1 
                self.w += (alpha**2)*(q_td_error)*self.e
                s_t = s_t_1
                a_t = a_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                temperature = global_time_step**(1.0/reduction_factor)
                # temperature = temperature / (global_time_step + 2)
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)        

class Qlearning(TD):
   
    def __init__(self, mdp, fourier_order, epsilon, alpha, num_training_episodes):
        super(Qlearning, self).__init__(mdp, alpha=alpha, order=fourier_order)
        self.episodes = num_training_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / 1
        q_a_1, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['forward'])
        q_a_2, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['neutral'])
        q_a_3, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['reverse'])
        q_values = [q_a_1, q_a_2, q_a_3]
        if self.debug:
            print "QFOR: ", q_values[0], " QNEUT: ", q_values[1], "QREV: ", q_values[2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            return argmax[0]
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            return coin_toss
    
    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        
        self.initialize_weights()
        self.debug = debug

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            mse = 0
            time_step = 0
            temperature = 1
            g = 0
            while not self.mdp.is_terminal_state(s_t) and time_step <= 1000:
                alpha = alpha / temperature
                a_t = self.epsilon_greedy_action_selection(s_t)
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                q_s_prime_a1, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['forward'])
                q_s_prime_a2, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['neutral'])
                q_s_prime_a3, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['reverse'])
                q_s_a, dq_dw = self.mdp.get_q_value_function(self.order, self.w, s_t, a_t)
                q_td_error = ((r_t + self.gamma*(max(q_s_prime_a1, q_s_prime_a2, q_s_prime_a3)) - q_s_a)*dq_dw)
                self.w += alpha*(q_td_error)
                s_t = s_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)

class QLambdaAC(TD):
   
    def __init__(self, mdp, fourier_order, epsilon, alpha, num_training_episodes):
        super(QLambdaAC, self).__init__(mdp, alpha=alpha, order=fourier_order)
        self.episodes = num_training_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma
        print "QLAMBDA AC"

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / 1
        q_a_1, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['forward'])
        q_a_2, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['neutral'])
        q_a_3, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['reverse'])
        q_values = [q_a_1, q_a_2, q_a_3]
        if self.debug:
            print "QFOR: ", q_values[0], " QNEUT: ", q_values[1], "QREV: ", q_values[2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            return argmax[0]
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            return coin_toss
    
    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        
        self.initialize_weights()
        self.debug = debug
        lamb = 0.9

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            mse = 0
            time_step = 0
            temperature = 1
            g = 0
            e1 = self.initialize_weights_e_trace()
            e2 = self.initialize_weights_e_trace()
            while not self.mdp.is_terminal_state(s_t) and time_step <= 1000:
                alpha = alpha / temperature
                a_t = self.epsilon_greedy_action_selection(s_t)
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                q_s_prime_a1, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['forward'])
                q_s_prime_a2, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['neutral'])
                q_s_prime_a3, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['reverse'])
                q_s_a, dq_dw = self.mdp.get_q_value_function(self.order, self.w, s_t, a_t)
                q_td_error = ((r_t + self.gamma*(max(q_s_prime_a1, q_s_prime_a2, q_s_prime_a3)) - q_s_a))
                e1 = lamb*self.gamma*e1 + dq_dw
                self.w += alpha*(q_td_error)*self.e
                s_t = s_t_1
                e2 = lamb*self.gamma*e2 + 1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)

class QLambda(TD):
   
    def __init__(self, mdp, fourier_order, epsilon, alpha, num_training_episodes):
        super(QLambda, self).__init__(mdp, alpha=alpha, order=fourier_order)
        self.episodes = num_training_episodes
        self.epsilon = epsilon
        self.gamma = self.mdp.gamma
        print "QLAMBDA"

    def epsilon_greedy_action_selection(self, state, temperature=1):
        random_number = 1.0*random.randint(0,99)/100
        epsilon = self.epsilon / 1
        q_a_1, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['forward'])
        q_a_2, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['neutral'])
        q_a_3, _ = self.mdp.get_q_value_function(self.order, self.w, state, self.mdp.actions['reverse'])
        q_values = [q_a_1, q_a_2, q_a_3]
        if self.debug:
            print "QFOR: ", q_values[0], " QNEUT: ", q_values[1], "QREV: ", q_values[2]
        if 1 - epsilon >= random_number:
            '''Select uniformly from the set of values argmax(q_values[s, ... ])'''
            arg_max = np.argwhere(q_values == np.amax(q_values))
            coin_toss = random.randint(0, len(arg_max) - 1)
            argmax = arg_max[coin_toss]
            return argmax[0]
        else:
            '''Select uniformly randomly from the set of actions'''
            coin_toss = random.randint(0, len(q_values) - 1)
            return coin_toss
    
    def learn(self, reduction_factor=4, plot=False, debug=False):
        X, y = [], []
        X_ep, y_ep = [], []
        global_time_step, time_step = 0, 0
        alpha = self.alpha
        temperature = 1
        
        self.initialize_weights()
        self.debug = debug
        lamb = 0.9

        for episode in range(self.episodes):
            if debug:
                print "------------------------------"
                print "AT EPISODE: ", episode + 1
            s_t = self.mdp.get_init_state()
            mse = 0
            time_step = 0
            temperature = 1
            g = 0
            self.initialize_weights_e_trace()
            while not self.mdp.is_terminal_state(s_t) and time_step <= 1000:
                alpha = alpha / temperature
                a_t = self.epsilon_greedy_action_selection(s_t)
                s_t_1 = self.mdp.transition_function(s_t, a_t)
                r_t = self.mdp.reward_function(s_t, a_t, s_t_1)
                q_s_prime_a1, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['forward'])
                q_s_prime_a2, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['neutral'])
                q_s_prime_a3, _ = self.mdp.get_q_value_function(self.order, self.w, s_t_1, self.mdp.actions['reverse'])
                q_s_a, dq_dw = self.mdp.get_q_value_function(self.order, self.w, s_t, a_t)
                q_td_error = ((r_t + self.gamma*(max(q_s_prime_a1, q_s_prime_a2, q_s_prime_a3)) - q_s_a))
                self.e = lamb*self.gamma*self.e + dq_dw
                self.w += alpha*(q_td_error)*self.e
                s_t = s_t_1
                g = g + r_t*(self.gamma**time_step)
                global_time_step += 1
                time_step += 1
                mse += q_td_error**2
                temperature = global_time_step**(1.0/reduction_factor)
            mse = mse / time_step
            X_ep.append(episode)
            y_ep.append(g)
            if debug:
                print "Return:  ", g
                print "------------------------------"
        if plot:
            plt.plot(X_ep, y_ep)
            plt.show()
        return X_ep, y_ep, np.sum(np.array(y_ep))*1.0/len(y_ep)

if __name__ == "__main__":
    fourier_order = 3
    mdp = MDP(order=fourier_order)
    td = TD(mdp, num_training_episodes=100, order=fourier_order)
    num_trials = 50
    num_training_episodes = 100
    
    hyperparam_search = False
    switch_sarsa = 1
    X = np.arange(num_training_episodes)
    Y = []

    if switch_sarsa == 0:
        print "------------" 
        print "SARSA" 
        print "------------"
    elif switch_sarsa == 1: 
        print "------------"
        print "Q-LEARNING"
        print "------------"
    elif switch_sarsa == 2:
        print "------------" 
        print "SARSA NN" 
        print "------------"
    elif switch_sarsa == 3:
        print "------------"
        print "Q-LEARNING NN"
        print "------------"

    # if hyperparam_search:
    #     '''HyperParameter Search'''
    #     alphas = get_hyperparams(range_of_param=[1e-3, 1e-1], interval=10, multiplicative=True)
    #     epsilons = get_hyperparams(range_of_param=[1e-2, 1e-1], interval=0.02, multiplicative=False)
    #     reduction_factors = get_hyperparams(range_of_param=[3,10], interval=1, multiplicative=False)
    #     G = -2**31
    #     params = []
    #     for alpha in alphas:
    #         for epsilon in epsilons:
    #             for reduction_factor in reduction_factors:
    #                 print "RETURN for alpha", str(alpha), " epsilon ", str(epsilon), " reductionFactor ", str(reduction_factor), " : "
    #                 if switch_sarsa == 0:
    #                     sarsa = Sarsa(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
    #                     _, y, g = sarsa.learn(reduction_factor=reduction_factor, plot=False, debug=False)
    #                 elif switch_sarsa == 1:
    #                     qlearn = Qlearning(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
    #                     _, y, g = qlearn.learn(reduction_factor=reduction_factor, plot=False, debug=False)
    #                 elif switch_sarsa == 2:
    #                     sarsa = Sarsa_NN(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
    #                     _, y, g = sarsa.learn(reduction_factor=reduction_factor, plot=False, debug=False)
    #                 elif switch_sarsa == 3:
    #                     qlearn = Qlearning_NN(mdp, epsilon=epsilon, alpha=alpha, train_episodes=num_training_episodes)
    #                     _, y, g = qlearn.learn(reduction_factor=reduction_factor, plot=False, debug=False)
    #                 print g
    #                 if G < g:
    #                     G = g
    #                     params = [alpha, epsilon, reduction_factor]
    #                     print "BEST PARAMS: "
    #                     print params
   
    # if not hyperparam_search:
    #     #alpha, epsilon, reduction_factor: alpha = alpha/(temp**red_fac)
    #     params = [1e-1, 1e-1, 2]

    params = [1e-5, 1e-4, 0.05]

    for trial in range(num_trials):
        print "AT TRIAL: ", trial + 1
        if switch_sarsa == 0:
            sarsa = SarsaLambda(mdp, fourier_order=fourier_order, epsilon=params[1], alpha=params[0], num_training_episodes=num_training_episodes)
            _, y, _ = sarsa.learn(reduction_factor=params[2], plot=False, debug=False)
        elif switch_sarsa == 1:
            qlearn = QLambdaAC(mdp, fourier_order=fourier_order, epsilon=params[1], alpha=params[0], num_training_episodes=num_training_episodes)
            _, y, _ = qlearn.learn(reduction_factor=params[2], plot=False, debug=False)
        Y.append(y)
    Y = np.array(Y)
    Y_mean = np.sum(Y, axis=0)
    Y_mean = Y_mean/num_trials
    Y_diff = np.repeat(Y_mean.reshape(1, num_training_episodes), num_trials, axis=0)    
    Y_diff = Y - Y_diff
    Y_diff = Y_diff ** 2
    Y_diff = np.sum(Y_diff, axis=0) / num_trials
    Y_diff = np.sqrt(Y_diff)
    plt.errorbar(X, Y_mean, yerr=Y_diff, fmt='o')
    
    if switch_sarsa == 0:
        print "------------" 
        print "SARSA" 
        print "------------"
    elif switch_sarsa == 1: 
        print "------------"
        print "Q-LEARNING"
        print "------------"
    elif switch_sarsa == 2:
        print "------------" 
        print "SARSA NN" 
        print "------------"
    elif switch_sarsa == 3:
        print "------------"
        print "Q-LEARNING NN"
        print "------------"

    plt.show()