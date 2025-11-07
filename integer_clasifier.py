import jax.numpy as np
import jax.random as random
import optimize as optimize
import projections as projections
rand_key = random.key(123)

## setup problem
xor_X = np.array([[0, 1, 0, 1],
                  [0, 0, 1, 1]])
xor_y = np.array([0, 1 , 1, 1])

optimizer = optimize.DouglassRachford

### define weight matrices
W0 = random.normal(rand_key,(3,2))
W0 = np.concatenate((W0,np.eye(3)),axis=1)  ## add bias row
W1 = random.normal(rand_key,4)
b0 = random.normal(rand_key,(3,))
b1 = 1


## set the value for positive class
delta = 1

error = []
import tools.integerProjections as inp
solver1 = inp.ClosestBilinearSolver(m=3,n=5)
solver2 = inp.ClosestBilinearSolverFixedX(m=1,n=4)
error = []
optInput = optimizer([],[projections.bilinearMatrix])
optActivation = optimizer([],[projections.sum_relu_proj])

optHidden = optimizer([], [solver2.solve])
optOutput= optimizer([lambda x,w,y :projections.classifierOutput(x,w,y,delta=delta)], [])
error = []
bias0 = []
bias1 = []
for iter in range(50):
    batchError = np.zeros(len(xor_y))
    for k, classSample in enumerate(xor_y):
        ## do a forward pass
        x_in = xor_X[:,k]
        x_in_aug = np.append(x_in, b0)
        x_hidden = W0 @ x_in_aug
        h_hidden =  x_hidden * (x_hidden > 0)  ## ReLU activation
        h_hidden_aug = np.append(h_hidden, b1)
        x_out = W1 @ h_hidden_aug
        ## calculate error
        if classSample==1:
            batchError = batchError.at[k].set((x_out - delta)**2 if x_out < delta else 0)
            #print("err",batchError)
        else:
            batchError = batchError.at[k].set((x_out)**2 if x_out > 0 else 0)
            #print("err",batchError)

        ## optimize output layer
        #print("xoux_hidden_augt original,", x_out)
        
        x_out, _, _ = optOutput.step_layer(x_out, np.eye(1), classSample)
        #print("xout", x_out, classSample)
        ## optimize hidden layer
        #print("xhidden original", x_hidden_aug, "w1 original", W1, "xout2", x_out)
        h_hidden_aug, w1_proj, x_out = optHidden.step_layer(h_hidden_aug, np.array([W1]),np.array([x_out]) )
        w1_proj= w1_proj[0]
        x_out = x_out[0]
        h_hidden= h_hidden_aug[:][:-1]
        #print("xhidden", x_hidden_aug, "w1 proj", w1_proj, "xout", x_out)
        
        ## optimize input layer
        #print(W0@x_in_aug)
        x_hidden, _,  h_hidden= optActivation.step_layer(x_hidden, np.eye(len(x_hidden)), h_hidden)
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
print("minimal  error", min(error))

