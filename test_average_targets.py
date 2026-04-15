import os
import torch

# This test requires PyTorch to be compiled with our new input_buffer.cpp modification.
# We set the environment variable that triggers the new averaging logic.
os.environ["PTORCH_PROJECTION_AVERAGE_TARGETS"] = "1"

def test_average_targets():
    # Create a simple tensor that requires gradient
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    # We use x in 3 different pathways (3 consumers of x).
    # Typically, PyTorch would sum the gradients (x.grad = 3 * target).
    # With the new behavior, it should average them (x.grad = target).
    out1 = x * 2.0
    out2 = x * 2.0
    out3 = x * 2.0

    # Total loss is the sum of the outputs.
    # For a standard sum, the gradient w.r.t x would be:
    # d(out1)/dx = 2, d(out2)/dx = 2, d(out3)/dx = 2
    # Standard PyTorch sum: x.grad = 2 + 2 + 2 = 6
    # Averaged (our target propagation logic): x.grad = (2 + 2 + 2) / 3 = 2
    loss = out1.sum() + out2.sum() + out3.sum()
    loss.backward()

    print(f"Gradient with PTORCH_PROJECTION_AVERAGE_TARGETS=1: {x.grad}")
    
    # We expect the gradient to be [2.0, 2.0] if the averaging works.
    # Otherwise, it would be [6.0, 6.0] if it just sums.
    if torch.allclose(x.grad, torch.tensor([2.0, 2.0])):
        print("Success! Gradients were averaged across the 3 usages.")
    elif torch.allclose(x.grad, torch.tensor([6.0, 6.0])):
        print("Failed! Gradients were summed. (Did you recompile PyTorch?)")
    else:
        print("Unexpected result.")

if __name__ == "__main__":
    test_average_targets()
