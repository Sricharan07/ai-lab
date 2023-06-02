from hmmlearn import hmm
import numpy as np

observations = np.array([[1], [2], [3], [4]]) 
hidden_states = np.array([0, 1, 0, 1])  

model = hmm.MultinomialHMM(n_components=2)  

model.fit(observations, lengths=[len(observations)])

print("Initial probabilities:")
print(model.startprob_)
print("\nTransition matrix:")
print(model.transmat_)
print("\nEmission matrix:")
print(model.emissionprob_)
