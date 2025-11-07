# %%
import jax.numpy as np
import jax.random as random
import optimize
import projections
rand_key = random.key(123)

# %%
## setup problem
inputDim = 2
hiddenDim = 3
outputDim = 1
samples = 2**inputDim

inputData = random.choice(rand_key, np.array([0,1]), shape=(inputDim, samples))
outputData = random.choice(rand_key, np.array([0,1]), shape=(outputDim, samples))

optimizer = optimize.DouglassRachford

# %%
### define weight matrices
W0 = random.normal(rand_key,(hiddenDim,inputDim))
W0 = np.concatenate((W0,np.eye(hiddenDim)),axis=1)  ## add bias row
W1 = random.normal(rand_key,(outputDim, hiddenDim+1))
b0 = random.normal(rand_key,(hiddenDim,))
b1 = inputDim

## set the value for positive class
delta = 1

# %%
## implementation using alternating Projections
error = []
optInput = optimizer([projections.stepActivation, projections.orthonormalizeMatrix],[projections.bilinearMatrix])
optHidden = optimizer([], [projections.bilinearMatrix])
optOutput= optimizer([lambda x,w,y :projections.classifierOutputVector(x,w,y,delta=delta)], [])
error = []
bias0 = []
bias1 = []
for iter in range(1000):
    batchError = np.zeros(len(outputData))
    for k, sample in enumerate(outputData.T):
        ## do a forward pass
        x_in = inputData[:,k]
        x_in_aug = np.append(x_in, b0)
        x_hidden = W0 @ x_in_aug
        h_hidden = np.where(x_hidden >= 0, 1.0, 0.0)  # perform step activation
        h_hidden_aug = np.append(h_hidden, b1)
        x_out = W1 @ h_hidden_aug
        
        ## calculate error
        for i, classSubSample in enumerate(sample):
            if classSubSample==1:
                batchError = batchError.at[k].set(batchError[k] + (x_out[i] - delta)**2 if x_out[i] < delta else 0)
            else:
                batchError = batchError.at[k].set(batchError[k] + (x_out[i])**2 if x_out[i] > 0 else 0)

        x_out, _, _ = optOutput.step_layer(x_out, np.eye(outputDim), sample)
        h_hidden_aug, w1_proj, x_out = optHidden.step_layer(h_hidden_aug, W1, x_out)
        h_hidden= h_hidden_aug[:][:-1]

        x_in_aug, w0_proj, x_hidden = optInput.step_layer(x_in_aug, W0, x_hidden)

        b1 = b1 + (1/(k+1))*(h_hidden_aug[:][-1:] - b1)
        b0 = b0 + (1/(k+1))*(x_in_aug[:][-len(b0):] - b0) 
        bias0.append(b0)
        
        W0 = W0 + (1/(k+1))*(w0_proj - W0)
        W1 = W1 + (1/(k+1))*(w1_proj - W1)
    error.append(np.mean(batchError))
    print(iter)

import matplotlib.pyplot as plt
plt.plot(error)
plt.show()
print("final error", error[-1])

# %%
