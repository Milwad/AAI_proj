import torch
from config import Config

class Decoder(torch.nn.Module):

    def __init__(self, latent_dim : int, hidden_dims : list[int] | tuple[int, ...], output_dim : int, activation : type[torch.nn.Module] = torch.nn.ReLU) -> None:
        
        super().__init__()
        dims = [latent_dim] + list(reversed(hidden_dims))

        layers: list[torch.nn.Module] = []

        for d_in,d_out in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(in_features=d_in, out_features=d_out))
            layers.append(activation())
            # layer += [torch.nn.Linear(in_features=d_in, out_features=d_out), activation()]
        
        layers += [torch.nn.Linear(in_features=dims[-1], out_features=output_dim)] # final output layer
        # unpack layers into an sequential neural net
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z) -> torch.Tensor:

        return self.net(z)

if __name__ == "__main__":
    conf = Config()
    torch.manual_seed(conf.seed)
    latent_dim, hidden_dims = conf.latent_dim, conf.hidden_dims
    n_features = 29  # V1..V28 + Amount; dropping time
    dec = Decoder(latent_dim=latent_dim, hidden_dims=hidden_dims, output_dim=n_features, activation=conf.activation)


    # Shape
    print("[1]", dec)
    last = list(dec.modules())[-1]
    print(f"last module: {last}")

    # Structure
    z = torch.randn(256, latent_dim)
    x_hat : torch.Tensor = dec.forward(z)
    print(f"[2] output: {tuple(x_hat.shape)} {x_hat.dtype} (expected: (256, {n_features}) torch.float32)")

    # Parameter count
    dims = [latent_dim, *reversed(hidden_dims), n_features]
    expected = sum(d_in * d_out + d_out for d_in, d_out in zip(dims[:-1], dims[1:]))
    
    # number of elements
    actual = sum(p.numel() for p in dec.parameters())
    print(f"[3] paramater count: {actual} (expected: {expected})")

    # Unbounded output
    print(f"[4] output range: {x_hat.min():.3f} - {x_hat.max():.3f}")

    # Gradients flow: call out.sum().backward(), then check that every parameter's .grad is not None.
    x_hat.sum().backward()
    missing = [n for n, p in dec.named_parameters() if p.grad is None]
    zero = [n for n, p in dec.named_parameters() if p.grad is not None and p.grad.abs().sum() == 0]
    print(f"[5] Missing params: {missing or 'none'}, zerod params: {zero or 'none'} (expect none and none)")

    # Configurable
    dec2 = Decoder(latent_dim=4, hidden_dims=(128,64,32), output_dim=31, activation=torch.nn.LeakyReLU)
    out2 = dec2(torch.randn(10, 4))
    print(f"[6] dec2 output: {tuple(out2.shape)} (expected: (10, 31)) | activation: {dec2.net[1]} (expected: LeakyReLU)")
    print(f"    single sample: {tuple(dec(torch.randn(1, latent_dim)).shape)} (expected: (1, {n_features}))")
