import torch

from models import LegalAutoencoder


def test_encode_matches_forward_latent_features():
    torch.manual_seed(7)
    model = LegalAutoencoder(vocab_size=32, embed_dim=8, hidden_dim=12)
    model.eval()
    inputs = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        encoded = model.encode(inputs)
        _, forward_latent = model(inputs)

    assert torch.allclose(encoded, forward_latent)
