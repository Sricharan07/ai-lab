import numpy as np
from hmmlearn import hmm

obs_seq = np.array([[1], [0], [1], [1], [0]])

initial_prob = np.array([0.6, 0.4])
transition_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_prob = np.array([[0.3, 0.7], [0.8, 0.2]])

model = hmm.GaussianHMM(n_components=2)
model.startprob_ = initial_prob
model.transmat_ = transition_prob
model.means_ = emission_prob

model.fit(obs_seq)

learned_initial_prob = model.startprob_
learned_transition_prob = model.transmat_
learned_emission_prob = model.means_

print("Learned Initial Probabilities:")
print(learned_initial_prob)
print("Learned Transition Probabilities:")
print(learned_transition_prob)
print("Learned Emission Probabilities:")
print(learned_emission_prob)
