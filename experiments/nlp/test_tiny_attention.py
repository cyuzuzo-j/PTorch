"""
Test Suite: TinyAttention (ptorch)

Verifies that the TinyAttention model — which uses Conversion, MultiHeadAttention,
and LinearBias projection layers — produces correct shapes, converges on a simple
synthetic task, and behaves deterministically.

Run:
    python experiments/nlp/test_tiny_attention.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import torch
import torch.nn as tnn
import torch.nn.functional as F
from ptorch.nn.modules import LinearBias, Linear, ReLU, MultiHeadAttention, Conversion
from ptorch.core.ops import CrossEntropyProjection
import ptorch.optim_static as ptorch_optim


# ── Model (copied from bench_ptorch.py) ─────────────────────
class TinyAttention(tnn.Module):
    def __init__(self, vocab_size, embed_dim, classes):
        super().__init__()
        self.embedding = tnn.Embedding(vocab_size, embed_dim)
        self.conversion = Conversion()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, heads=1)
        self.out = LinearBias(embed_dim, classes)
        self.embed_dim = embed_dim

    def forward(self, x):
        embedded = self.embedding(x)           # B, S, E
        embedded = self.conversion(embedded)   # bridge: gradient → projection
        context = self.attention(embedded)     # B, S, E
        pooled = context.mean(dim=1)           # B, E
        return self.out(pooled)


# ── Stripped-down attention model (no embedding/conversion) ───
class AttentionOnly(tnn.Module):
    """MultiHeadAttention + LinearBias, takes float features directly."""
    def __init__(self, embed_dim, classes, heads=1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, heads=heads)
        self.out = LinearBias(embed_dim, classes)

    def forward(self, x):
        context = self.attention(x)
        pooled = context.mean(dim=1)
        return self.out(pooled)


# ── Synthetic data helpers ───────────────────────────────────
def make_token_data(n, vocab_size, seq_len, classes=2, seed=42):
    """Synthetic binary classification: label = majority token parity."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n, seq_len), generator=g)
    labels = (tokens.float().mean(dim=1) > vocab_size / 2).long()
    return tokens, labels


def make_float_data(n, embed_dim, seq_len, classes=2, seed=42):
    """Synthetic float sequences with a learnable signal."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, seq_len, embed_dim, generator=g)
    # Label based on sign of mean feature across sequence
    labels = (x.mean(dim=(1, 2)) > 0).long()
    return x, labels


# ── Training utilities ───────────────────────────────────────
def train_step(model, x, y, optimizer, classes):
    """Single projection-based training step."""
    logits = model(x)
    y_oh = F.one_hot(y.long(), num_classes=classes).float()
    projected = CrossEntropyProjection.apply(logits, y_oh)
    optimizer.zero_grad()
    projected.sum().backward()
    loss = F.cross_entropy(logits.detach(), y.long())
    optimizer.step()
    return loss.item()


def compute_accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=-1)
    model.train()
    return (preds == y).float().mean().item()


# ══════════════════════════════════════════════════════════════
#  TEST 1: Forward Shape
# ══════════════════════════════════════════════════════════════
def test_forward_shape():
    print("=" * 60)
    print("TEST 1: Forward Shape")
    print("=" * 60)

    torch.manual_seed(0)
    vocab_size, embed_dim, classes = 100, 32, 2
    model = TinyAttention(vocab_size, embed_dim, classes)

    for batch, seq_len in [(1, 5), (4, 10), (8, 20)]:
        x = torch.randint(0, vocab_size, (batch, seq_len))
        out = model(x)
        expected = (batch, classes)
        assert out.shape == expected, f"Shape mismatch: {out.shape} != {expected}"
        print(f"  batch={batch}, seq_len={seq_len} → {out.shape}  ✓")

    print("  ✓ PASSED\n")


# ══════════════════════════════════════════════════════════════
#  TEST 2: Backward Shape
# ══════════════════════════════════════════════════════════════
def test_backward_shape():
    print("=" * 60)
    print("TEST 2: Backward Shape (projection targets propagate)")
    print("=" * 60)

    torch.manual_seed(0)
    vocab_size, embed_dim, classes = 100, 32, 2
    model = TinyAttention(vocab_size, embed_dim, classes)

    x = torch.randint(0, vocab_size, (4, 10))
    y = torch.randint(0, classes, (4,))

    logits = model(x)
    y_oh = F.one_hot(y, num_classes=classes).float()
    projected = CrossEntropyProjection.apply(logits, y_oh)
    projected.sum().backward()

    # Check that gradients arrive at all parameters
    n_with_grad = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += 1
        has_grad = p.grad is not None
        if has_grad:
            n_with_grad += 1
            assert p.grad.shape == p.shape, f"Shape mismatch for {name}"
        print(f"  {name:40s}  shape={str(p.shape):20s}  grad={'✓' if has_grad else '✗'}")

    # Embedding gets gradient via Conversion bridge
    assert model.embedding.weight.grad is not None, "Embedding should receive gradient via Conversion"
    # All ptorch params should have grad
    assert n_with_grad == n_total, f"Only {n_with_grad}/{n_total} params received gradients"
    print(f"\n  All {n_total} parameters received gradients  ✓")
    print("  ✓ PASSED\n")


# ══════════════════════════════════════════════════════════════
#  TEST 3: Component Convergence (AttentionOnly — no embedding)
# ══════════════════════════════════════════════════════════════
def test_component_convergence():
    print("=" * 60)
    print("TEST 3: Component Convergence (AttentionOnly)")
    print("=" * 60)

    torch.manual_seed(42)
    embed_dim, classes, seq_len = 32, 2, 8
    n_train = 256

    x, y = make_float_data(n_train, embed_dim, seq_len, classes, seed=42)
    model = AttentionOnly(embed_dim, classes, heads=1)
    optimizer = ptorch_optim.ProjectionAdadelta(model.parameters(), lr=10)

    n_steps = 150
    batch_size = 64
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, n_train, (batch_size,))
        loss = train_step(model, x[idx], y[idx], optimizer, classes)
        losses.append(loss)
        if step % 30 == 0:
            print(f"  Step {step:3d}  loss = {loss:.4f}")

    initial_avg = sum(losses[:10]) / 10
    final_avg = sum(losses[-10:]) / 10
    improvement = (initial_avg - final_avg) / initial_avg * 100

    acc = compute_accuracy(model, x, y)
    print(f"\n  Loss: {initial_avg:.4f} → {final_avg:.4f}  ({improvement:+.1f}%)")
    print(f"  Training acc: {acc:.2%}")

    assert final_avg < initial_avg, f"Loss did not decrease: {initial_avg:.4f} → {final_avg:.4f}"
    print("  ✓ PASSED\n")


# ══════════════════════════════════════════════════════════════
#  TEST 4: Full TinyAttention Convergence
# ══════════════════════════════════════════════════════════════
def test_full_convergence():
    print("=" * 60)
    print("TEST 4: Full TinyAttention Convergence")
    print("=" * 60)

    torch.manual_seed(42)
    vocab_size, embed_dim, classes, seq_len = 200, 32, 2, 12
    n_train = 400

    tokens, labels = make_token_data(n_train, vocab_size, seq_len, classes, seed=42)
    model = TinyAttention(vocab_size, embed_dim, classes)

    # Split params: embedding uses Adam, ptorch params use projection optimizer
    embedding_params = list(model.embedding.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    proj_params = [p for p in model.parameters() if id(p) not in embedding_ids]

    embedding_opt = torch.optim.Adam(embedding_params, lr=0.005)
    proj_opt = ptorch_optim.ProjectionAdadelta(proj_params, lr=10)

    n_steps = 200
    batch_size = 64
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, n_train, (batch_size,))
        xb, yb = tokens[idx], labels[idx]

        logits = model(xb)
        y_oh = F.one_hot(yb.long(), num_classes=classes).float()
        projected = CrossEntropyProjection.apply(logits, y_oh)

        embedding_opt.zero_grad()
        proj_opt.zero_grad()
        projected.sum().backward()

        loss = F.cross_entropy(logits.detach(), yb.long()).item()
        embedding_opt.step()
        proj_opt.step()

        losses.append(loss)
        if step % 40 == 0:
            acc = compute_accuracy(model, tokens, labels)
            print(f"  Step {step:3d}  loss = {loss:.4f}  acc = {acc:.2%}")

    initial_avg = sum(losses[:10]) / 10
    final_avg = sum(losses[-10:]) / 10
    improvement = (initial_avg - final_avg) / initial_avg * 100
    final_acc = compute_accuracy(model, tokens, labels)

    print(f"\n  Loss: {initial_avg:.4f} → {final_avg:.4f}  ({improvement:+.1f}%)")
    print(f"  Final training acc: {final_acc:.2%}")

    assert final_avg < initial_avg, f"Loss did not decrease: {initial_avg:.4f} → {final_avg:.4f}"
    assert final_acc > 0.55, f"Accuracy too low: {final_acc:.2%} (expected > 55%)"
    print("  ✓ PASSED\n")


# ══════════════════════════════════════════════════════════════
#  TEST 5: Optimizer Sweep
# ══════════════════════════════════════════════════════════════
def test_optimizer_sweep():
    print("=" * 60)
    print("TEST 5: Optimizer Sweep")
    print("=" * 60)

    optimizers = {
        "AlternatingProjections":  lambda p: ptorch_optim.AlternatingProjections(p),
        "ProjectionAdadelta":      lambda p: ptorch_optim.ProjectionAdadelta(p, lr=10),
        "ProjectionSGD":           lambda p: ptorch_optim.ProjectionSGD(p, lr=1.0),
    }

    embed_dim, classes, seq_len = 32, 2, 8
    n_train, n_steps, batch_size = 256, 100, 64

    for name, make_opt in optimizers.items():
        torch.manual_seed(42)
        x, y = make_float_data(n_train, embed_dim, seq_len, classes, seed=42)
        model = AttentionOnly(embed_dim, classes, heads=1)
        opt = make_opt(model.parameters())

        losses = []
        for step in range(n_steps):
            idx = torch.randint(0, n_train, (batch_size,))
            loss = train_step(model, x[idx], y[idx], opt, classes)
            losses.append(loss)

        initial_avg = sum(losses[:10]) / 10
        final_avg = sum(losses[-10:]) / 10
        converged = final_avg < initial_avg
        status = "✓" if converged else "✗"
        print(f"  {status} {name:30s}  {initial_avg:.4f} → {final_avg:.4f}")
        assert converged, f"{name} did not converge"

    print("  ✓ PASSED\n")


# ══════════════════════════════════════════════════════════════
#  TEST 6: Determinism
# ══════════════════════════════════════════════════════════════
def test_determinism():
    print("=" * 60)
    print("TEST 6: Determinism (same seed → same losses)")
    print("=" * 60)

    embed_dim, classes, seq_len = 32, 2, 8
    n_train, n_steps, batch_size = 128, 30, 32

    all_losses = []
    for run in range(2):
        torch.manual_seed(123)
        x, y = make_float_data(n_train, embed_dim, seq_len, classes, seed=123)
        model = AttentionOnly(embed_dim, classes, heads=1)
        opt = ptorch_optim.ProjectionAdadelta(model.parameters(), lr=10)

        losses = []
        for step in range(n_steps):
            torch.manual_seed(step + 1000)  # deterministic batch sampling
            idx = torch.randint(0, n_train, (batch_size,))
            loss = train_step(model, x[idx], y[idx], opt, classes)
            losses.append(loss)
        all_losses.append(losses)

    max_diff = max(abs(a - b) for a, b in zip(all_losses[0], all_losses[1]))
    print(f"  Max loss difference between runs: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Non-deterministic: max_diff = {max_diff}"
    print("  ✓ PASSED\n")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    test_forward_shape()
    test_backward_shape()
    test_component_convergence()
    test_full_convergence()
    test_optimizer_sweep()
    test_determinism()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
