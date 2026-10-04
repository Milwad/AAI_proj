import torch
from config import Config


class Encoder(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list[int] | tuple[int, ...], latent_dim: int, activation: type[torch.nn.Module] = torch.nn.ReLU) -> None:

        super().__init__()
        dims: list[int] = [input_dim] + list(hidden_dims)

        layers: list[torch.nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(in_features=d_in, out_features=d_out))
            layers.append(activation())

        self.trunk: torch.nn.Sequential = torch.nn.Sequential(*layers)

        self.mean_head: torch.nn.Linear = torch.nn.Linear(in_features=dims[-1], out_features=latent_dim)
        self.logvar_head: torch.nn.Linear = torch.nn.Linear(in_features=dims[-1], out_features=latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # x: [B, input_dim] -> h: [B, hidden_dims[-1]]
        h: torch.Tensor = self.trunk(x)

        # Latent Gaussian parameters: each [B, latent_dim]
        mean: torch.Tensor = self.mean_head(h)
        logvar: torch.Tensor = self.logvar_head(h)

        return mean, logvar


if __name__ == "__main__":
    
    conf = Config()
    torch.manual_seed(conf.seed)

    n_features = 29  # V1..V28 + Amount; dropping time
    hidden_dims = conf.hidden_dims
    latent_dim = conf.latent_dim

    enc = Encoder(n_features, hidden_dims, latent_dim)  # default activation: ReLU

    # Structure, trunk ends with an activation, two separate heads without one

    print("[1]", enc)

    # Output shapes and dtypes
    x = torch.randn(256, n_features)
    mean, logvar = enc(x)
    print(f"[2] mean: {tuple(mean.shape)} {mean.dtype} | logvar: {tuple(logvar.shape)} {logvar.dtype} "
          f"(expect (256, {latent_dim}) torch.float32 for both)")
    
    # Parameter count trunk + two heads (weights AND biases)
    dims = [n_features, *hidden_dims]
    trunk = sum(d_in * d_out + d_out for d_in, d_out in zip(dims[:-1], dims[1:]))
    heads = 2 * (dims[-1] * latent_dim + latent_dim)
    actual = sum(p.numel() for p in enc.parameters())
    print(f"[3] params: {actual:,} (expect {trunk + heads:,})")

    # Heads are independent, unbounded and sigma is valid
    sigma = torch.exp(0.5 * logvar)
    print(f"[4] mean == logvar: {torch.equal(mean, logvar)} (expect False)")
    print(f"    logvar range: {logvar.min():.3f} to {logvar.max():.3f} (expect values around 0, some negative)")
    print(f"    sigma range: {sigma.min():.3f} to {sigma.max():.3f} | all positive & finite: "
          f"{bool((sigma > 0).all() and torch.isfinite(sigma).all())} (expect ~1, True)")
    
    # Gradients flow
    (mean.sum() + logvar.sum()).backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    zero = [n for n, p in enc.named_parameters() if p.grad is not None and p.grad.abs().sum() == 0]
    print(f"[5] params without grad: {missing or 'none'} | all-zero grads: {zero or 'none'} (expect none, none)")

    # Deterministic and batch-independent
    with torch.no_grad():
        mean_single, logvar_single = enc(x[:1])
    print(f"[6] single == in-batch: {torch.allclose(mean_single, mean[:1]) and torch.allclose(logvar_single, logvar[:1])} (expect True)")

    # Configurable
    enc2 = Encoder(31, (128, 64, 32), 4, torch.nn.LeakyReLU)
    mean2, logvar2 = enc2(torch.randn(10, 31))
    print(f"[7] alt config: mean {tuple(mean2.shape)}, logvar {tuple(logvar2.shape)} (expect (10, 4) for both)")