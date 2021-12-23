# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Discrete Hidden Markov Model (HMM) with scaling
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
from subprocess import run, PIPE

def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM:
    def __init__(self, M):
        self.M = M # number of hidden states
    
    def fit(self, X, max_iter=30):
        np.random.seed(123)
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        # determine V, the vocabulary size
        # assume observables are already integers from 0..V-1
        # X is a jagged array of observed sequences
        V = max(max(x) for x in X) + 1
        N = len(X)

        self.pi = np.ones(self.M) / self.M # initial state distribution
        self.A = random_normalized(self.M, self.M) # state transition matrix
        self.B = random_normalized(self.M, V) # output distribution

        # print("initial A:", self.A)
        # print("initial B:", self.B)

        costs = []
        for it in range(max_iter):
            if it % 10 == 0:
                print("it:", it)
            # alpha1 = np.zeros((N, self.M))
            alphas = []
            betas = []
            scales = []
            logP = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)
                scale = np.zeros(T)
                # alpha1[n] = self.pi*self.B[:,x[0]]
                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi*self.B[:,x[0]]
                scale[0] = alpha[0].sum()
                alpha[0] /= scale[0]
                for t in range(1, T):
                    alpha_t_prime = alpha[t-1].dot(self.A) * self.B[:, x[t]]
                    scale[t] = alpha_t_prime.sum()
                    alpha[t] = alpha_t_prime / scale[t]
                logP[n] = np.log(scale).sum()
                alphas.append(alpha)
                scales.append(scale)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in range(T - 2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1]) / scale[t+1]
                betas.append(beta)


            cost = np.sum(logP)
            costs.append(cost)

            # now re-estimate pi, A, B
            self.pi = sum((alphas[n][0] * betas[n][0]) for n in range(N)) / N

            den1 = np.zeros((self.M, 1))
            den2 = np.zeros((self.M, 1))
            a_num = np.zeros((self.M, self.M))
            b_num = np.zeros((self.M, V))
            for n in range(N):
                x = X[n]
                T = len(x)
                den1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T
                den2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T

                # numerator for A
                # a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num[i,j] += alphas[n][t,i] * betas[n][t+1,j] * self.A[i,j] * self.B[j, x[t+1]] / scales[n][t+1]
                # a_num += a_num_n

                # numerator for B
                # for i in range(self.M):
                #     for j in range(V):
                #         for t in range(T):
                #             if x[t] == j:
                #                 b_num[i,j] += alphas[n][t][i] * betas[n][t][i]
                for i in range(self.M):
                    for t in range(T):
                        b_num[i,x[t]] += alphas[n][t,i] * betas[n][t,i]
            self.A = a_num / den1
            self.B = b_num / den2
        # print("A:", self.A)
        # print("B:", self.B)
        # print("pi:", self.pi)

        # plt.plot(costs)
        # plt.show()

    def log_likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        scale = np.zeros(T)
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi*self.B[:,x[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha_t_prime = alpha[t-1].dot(self.A) * self.B[:, x[t]]
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        return np.log(scale).sum()

    def log_likelihood_multi(self, X):
        return np.array([self.log_likelihood(x) for x in X])

    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = np.log(self.pi) + np.log(self.B[:,x[0]])
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.B[j, x[t]])
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

def valid_conll_eval(fname):

	with open(fname, 'r') as file:
		data = file.read()

	pipe = run(["perl", "eval_conll2000_updated.pl"], stdout=PIPE, input=data, encoding='ascii')
	output = pipe.stdout

	tag_acc = float(output.split()[0])
	phrase_f1 = float(output.split()[1])

	print("tag_acc, phrase_f1", tag_acc, phrase_f1)
	return phrase_f1

def evaluate(predicted, target):
    filename = "temp.txt"
    f = open(filename, "w")
    for i in range(len(predicted)):
        if predicted[i] == 0:
            f.write("x y " + target[i] + " B\n")
        elif predicted[i] == 1:
            f.write("x y " + target[i] + " I\n")
        elif predicted[i] == 2:
            f.write("x y " + target[i] + " O\n")
        else:
            f.write("x y " + target[i] + " N\n")
    f.close()
    fscore = valid_conll_eval(filename)
    return fscore

def process_data():
    X = []
    x = []
    X_val = []
    x_val = []
    T_val = []
    t_val = []
    sequence_syms = dict()
    sequence = list()
    X_t = []
    x_t = []
    T = []
    t = []
    for line in open('../test.txt'):
        words = line.split()
        if len(words) == 0:
            X_t.append(x_t)
            x_t  = []
            T.append(t)
            t = []
            continue
        if words[0] not in sequence_syms:
            sequence_syms[words[0]] = len(sequence)
            sequence.append(words[0])
        x_t.append(sequence_syms[words[0]])
        t.append(words[2][0])
    
    for line in open('../validation.txt'):
        words = line.split()
        if len(words) == 0:
            X_val.append(x_val)
            x_val  = []
            T_val.append(t_val)
            t_val = []
            continue
        if words[0] not in sequence_syms:
            sequence_syms[words[0]] = len(sequence)
            sequence.append(words[0])
        x_val.append(sequence_syms[words[0]])
        t_val.append(words[2][0])

    for line in open('../train.txt'):
        words = line.split()
        if len(words) == 0:
            X.append(x)
            x  = []
            continue
        if words[0] not in sequence_syms:
            sequence_syms[words[0]] = len(sequence)
            sequence.append(words[0])
        x.append(sequence_syms[words[0]])
    return X, X_val, T_val, X_t, T

def run():
    X, X_val, T_val, X_t, T = process_data()

    num_states = [2, 3, 4]
    best_state_num = 0
    best_fscore = 0
    for n_state in num_states:
        hmm = HMM(n_state)
        hmm.fit(X)
        L = hmm.log_likelihood_multi(X).sum()
        print("LL with fitted params:", L)
        
        predicted = list()
        target = list()
        for i in range(len(X_val)):
            predicted.extend(hmm.get_state_sequence(X_val[i]))
            target.extend(T_val[i])

        fscore = evaluate(predicted, target)
        if fscore > best_fscore:
            best_fscore = fscore
            best_state_num = n_state
    
    print("Best number of hidden state:", best_state_num)
    hmm = HMM(best_state_num)
    hmm.fit(X + X_val)
    predicted = list()
    target = list()
    for i in range(len(X_t)):
        predicted.extend(hmm.get_state_sequence(X_t[i]))
        target.extend(T[i])

    evaluate(predicted, target)

if __name__ == '__main__':
    run()
