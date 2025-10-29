
import jax.numpy as np
import jax.random as random
import tools.optimize as optimize
import tools.projections as projections
rand_key = random.key(123)

xor_X = np.array([[0, 1, 0, 1],
                  [0, 0, 1, 1]])
xor_y = np.array([0, 1, 1, 1])

optimizer = optimize.AlternatingProjection

W0 = random.normal(rand_key,(2,2))*5
W0 = np.concatenate((W0,np.eye(2)),axis=1)  ## add bias row
W1 = random.normal(rand_key, 3)*5
b0 = random.normal(rand_key,(2,))*5
b1 = 1


delta = 1


error = []
import tools.integerProjections as inp
solver1 = inp.ClosestBilinearSolverY(m=2,n=4)
solver2 = inp.ClosestBilinearSolverY(m=1,n=3)
optInput = optimizer([projections.stepActivation],[solver1.solve])
optHidden = optimizer([], [solver2.solve])
optOutput= optimizer([lambda x,w,y :projections.classifierOutput(x,w,y,delta=delta)], [])
error = []
bias0 = []
bias1 = []
for iter in range(30):
    batchError = np.zeros(len(xor_y))
    for k, classSample in enumerate(xor_y):
        ## do a forward pass
        x_in = xor_X[:,k]
        x_in_aug = np.append(x_in, b0)
        x_hidden = W0 @ x_in_aug
        h_hidden = np.where(x_hidden >= 0, 1.0, 0.0)  # perform step activation
        h_hidden_aug = np.append(h_hidden, b1)
        x_out = W1 @ h_hidden_aug

        if classSample==1:
            print("xout", classSample, x_out, "input", x_in )
            batchError = batchError.at[k].set((x_out - delta)**2 if x_out < delta else 0)
        else:
            print("xout", classSample, x_out, "input", x_in, (x_out)**2 if x_out >= 0 else 0)
            batchError = batchError.at[k].set((x_out)**2 if x_out >= 0 else 0)


        x_out, _, _ = optOutput.step_layer(x_out, np.eye(1), classSample)

        h_hidden_aug, w1_proj, x_out = optHidden.step_layer(h_hidden_aug, np.array([W1]), np.array([x_out]))
        x_out = x_out[0]
        w1_proj = w1_proj[0]
        h_hidden= h_hidden_aug[:][:-1]
        #print("hidden opt",x_out, w1_proj @ h_hidden_aug )

        ## optimize input layer
        #print( "xin original", x_in_aug,"\n" "w0 original", W0,"\n" "xhidden original", x_hidden)
        #print(W0@x_in_aug)
        w0_proj = W0
        for i in range(5):
            x_in_aug, w0_proj, x_hidden = optInput.step_layer(x_in_aug, w0_proj, x_hidden)
        #print( "xin", x_in_aug, "w0 proj", w0_proj, "xhidden", x_hidden)

        b1 = b1 + (1/(k+1))*(h_hidden_aug[:][-1:] - b1)
        b0 = b0 + (1/(k+1))*(x_in_aug[:][-len(b0):] - b0) 
        bias0.append(b0)
        print(b0, b1)
        
        W0 = W0 + (1/(k+1))*(w0_proj - W0)
        W1 = W1 + (1/(k+1))*(w1_proj - W1)
    error.append(np.mean(batchError))
    print(iter)

import matplotlib.pyplot as plt
plt.plot(error)
print("final error", error[-1])



for k, classSample in enumerate(xor_y):
    print(b0, b1)
    ## do a forward pass
    x_in = xor_X[:,k]
    x_in_aug = np.append(x_in, b0)
    print("x_in_aug", x_in.shape, b0.shape)
    print( "x_in_aug", x_in_aug.shape, W0.shape)
    h_hidden = np.where(W0 @ x_in_aug >= 0, 1.0, 0.0)  # perform step activation
    x_hidden_aug = np.append(h_hidden, b1)
    x_out = W1 @ x_hidden_aug
    
    print("x_in", x_in, "x_hidden", x_hidden, "x_out", x_out, "class", classSample)



