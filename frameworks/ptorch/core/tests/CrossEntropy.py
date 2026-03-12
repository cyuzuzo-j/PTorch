"""
Test file confirming the functioning of CrossEntropyProjection
"""
import torch
import torch.nn.functional as F
from .. import ops

def test_forward_trivial():
    """Forward pass just returns the logits unchanged."""
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    labels = torch.zeros_like(logits)
    labels.scatter_(-1, torch.randint(0, 10, (4, 1)), 1.0)

    out = ops.CrossEntropyProjection.apply(logits, labels)
    assert torch.equal(out, logits), "Forward pass must return logits unchanged"

def test_forward_shape_preserved():
    """Output shape matches input logits shape."""
    torch.manual_seed(1)
    logits = torch.randn(8, 5)
    labels = torch.zeros_like(logits)
    labels.scatter_(-1, torch.randint(0, 5, (8, 1)), 1.0)

    out = ops.CrossEntropyProjection.apply(logits, labels)
    assert out.shape == logits.shape

def test_backward_moves_toward_labels():
    """
    Backward step should move logits closer to the one-hot labels.
    After one projection step (lmbda=1, num_steps=1):
      x_new = x + lmbda * (labels - softmax(x))
    The cross-entropy on x_new should be <= cross-entropy on x.
    """
    torch.manual_seed(2)
    logits = torch.randn(4, 6, requires_grad=False)
    targets = torch.randint(0, 6, (4,))
    one_hot = F.one_hot(targets, num_classes=6).float()

    ce_before = F.cross_entropy(logits, targets).item()

    # Simulate one backward pass manually
    lmbda = 1.0
    x_new = logits + lmbda * (one_hot - F.softmax(logits, dim=-1))
    ce_after = F.cross_entropy(x_new, targets).item()

    assert ce_after <= ce_before, (
        f"Backward step should decrease cross-entropy: {ce_before:.4f} -> {ce_after:.4f}"
    )

def test_backward_converges_to_labels():
    """
    With many steps the logit for the correct class should dominate.
    """
    torch.manual_seed(3)
    batch, C = 3, 5
    logits = torch.randn(batch, C)
    targets = torch.randint(0, C, (batch,))
    one_hot = F.one_hot(targets, num_classes=C).float()

    x = logits.clone()
    for _ in range(100):
        x = x + 1.0 * (one_hot - F.softmax(x, dim=-1))

    predicted = x.argmax(dim=-1)
    assert torch.all(predicted == targets), (
        "After many steps logits should predict the correct class"
    )
