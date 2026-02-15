# Benchmark to spot the difference between optimizers
The following test suite is designed to get a true view of each of the optimizers capaility.
Assesing the optimizers in the following scenario's.

# evading a local minimizers.
This test is designed to be very difficult for gradient based optimzers but easy for feasibility based ones.
Test setup is as follows:
- initialise with fully trained model using sgd on synthetic dataset that is right in a local minimizer.
- let the algorithm train for 1 epoch and see if it has stepped out and is going to the true minima or if it is stuck
- see what the influence of the relaxation parameter and the amount of steps per step is using the elser paper as reference
- Interesting algo's sgd vs dr vs dysktra

# general cifar 10 test
Proper benchmark evaluating 1 the influence of relaxation and also the difference between dysktra, my monumentum hack,.... only benchmarking the feasibility based models
- Similar to already tested benchmarks

# under-to-overparameterized
testing how the algorithpms converge when network complexity slowly increases from under to over parameterized
- use mnist and find online what the minimal complexity is
