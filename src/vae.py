from enc import Encoder
from dec import Decoder
import torch
import math
from pathlib import Path

class VAE(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list[int] | tuple[int, ...], latent_dim: int, activation: type[torch.nn.Module] = torch.nn.ReLU) -> None:

        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, activation=activation)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dims=hidden_dims, output_dim=input_dim, activation=activation)

    # trick to allow gradients to backprop, bring random element in not via parameters but via a random number(epsilon) adjusted to fit a normal dist
    def reparameterize(self, mean : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor:

        # we use 0.5 because its equivalent to squarerooting, c*ln(x) = ln(x^c)
        standard_dev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(standard_dev)

        return mean + epsilon * standard_dev

    def forward(self, x : torch.Tensor, sample : bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mean, logvar = self.encoder(x=x)
        z = self.reparameterize(mean=mean, logvar=logvar) if sample else mean
        x_hat = self.decoder(z=z)
        return x_hat, mean, logvar

ACTIVATION_MAP = {
    "ReLU": torch.nn.ReLU,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU": torch.nn.ELU,
    "GELU": torch.nn.GELU,
    "Tanh": torch.nn.Tanh,
    "Sigmoid": torch.nn.Sigmoid
}

def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:

    return torch.sum(torch.pow(x-x_hat, 2), dim=-1)

def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

    return -0.5 * torch.sum(1+logvar-torch.pow(mean,2) - torch.exp(logvar), dim=-1)

def vae_loss(x : torch.Tensor, x_hat : torch.Tensor, mean : torch.Tensor, logvar : torch.Tensor, beta : float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    reconstruction_loss = reconstruction_error(x=x, x_hat=x_hat)
    kl = kl_divergence(mean=mean, logvar= logvar)
    loss = reconstruction_loss + beta * kl

    return loss, reconstruction_loss, kl

def load_model(run_dir: str | Path) -> tuple[VAE, dict]:

    """Rebuilds and loads a model from run_dir/model.pt using its saved architecture."""
    run_path = Path(run_dir)
    ckpt_path = run_path / "model.pt" if run_path.is_dir() else run_path
    checkpoint = torch.load(ckpt_path, weights_only=False)
    cfg = checkpoint["config"]
    act_name = cfg.get("activation", "ReLU")
    activation = ACTIVATION_MAP.get(act_name, torch.nn.ReLU)
    model = VAE(
        input_dim=len(checkpoint["feature_names"]),
        hidden_dims=tuple(cfg["hidden_dims"]),
        latent_dim=cfg["latent_dim"],
        activation=activation,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint

  

if __name__ == "__main__":

    from config import Config
    from data import Data

    conf = Config()
    torch.manual_seed(conf.seed)
    n_features = 29  # V1..V28 + Amount ("drop"); 31 with "cyclic"
    latent_dim = conf.latent_dim
    beta = conf.beta

    vae = VAE(input_dim=n_features, hidden_dims=conf.hidden_dims, latent_dim=latent_dim, activation=conf.activation)

    # structure and parameter count
    print("[1] structure and param #, ", vae)
    n_enc = sum(p.numel() for p in vae.encoder.parameters())
    n_dec = sum(p.numel() for p in vae.decoder.parameters())
    n_all = sum(p.numel() for p in vae.parameters())
    print(f"params: encoder ({n_enc}) + decoder ({n_dec}), all = {n_all}")

    # forward shapes
    x = torch.randn(256, n_features)
    x_hat, mean, logvar = vae(x=x)
    print(f"[2] shape, x_hat = {tuple(x_hat.shape)}, mean = {tuple(mean.shape)}, logvar = {tuple(logvar.shape)}")

    # reparameterization check
    m = torch.full((100000, latent_dim), 2.0)
    lv = torch.full((100000, latent_dim), math.log(0.25))
    z = vae.reparameterize(mean=m, logvar=lv)
    print(f"[3] reparam. check, z mean = {z.mean()}, z std = {z.std()}")
    m_g = torch.zeros(4, latent_dim, requires_grad=True)
    lv_g = torch.zeros(4, latent_dim, requires_grad=True)
    vae.reparameterize(m_g, lv_g).sum().backward()
    print(f"dz/dmean all ones: {torch.equal(m_g.grad, torch.ones_like(m_g))}, logvar has grad: {lv_g.grad is not None}")

    # samppling behaviour check
    with torch.no_grad():
        # we use _ because we only need forwards x_hat, we do not care about the other outputs, and : means we are slicing from start:stop, 
        # where default start is 0 if nothing is specified
        a, _, _ = vae(x[:5])
        b, _, _ = vae(x[:5])
        c, _, _ = vae(x[:5], sample=False)
        d, _, _ = vae(x[:5], sample=False)

    # if you sample the output must be stochastic and if we dont sample the output must be deterministic
    print(f"[4] checking sample behaviour, we compare two sample=true: {torch.equal(a, b)}, and two sample=false: {torch.equal(c,d)}")

    # kl diverg check
    xz = torch.zeros(3, n_features)
    _, _, kl_prior = vae_loss(x=xz, x_hat=xz, mean=torch.zeros(3, latent_dim), logvar=torch.zeros(3, latent_dim), beta=beta)
    _, _, kl_mean1 = vae_loss(x=xz, x_hat=xz, mean=torch.ones(3, latent_dim), logvar=torch.zeros(3, latent_dim), beta=beta)
    _, _, kl_var4 = vae_loss(x=xz, x_hat=xz, mean=torch.zeros(3, latent_dim), logvar=torch.full((3, latent_dim), math.log(4.0)), beta=beta)

    print(f"[5] KL at prior: {kl_prior.tolist()}, expected: [0.0][0.0][0.0]")
    print(f" KL mean=1, var=1: {kl_mean1.tolist()}, expected: [{0.5 * latent_dim:.1f}][{0.5 * latent_dim:.1f}][{0.5 * latent_dim:.1f}]")
    print(f" KL mean=0, var=4: {kl_var4.tolist()}, expected: [{0.5 * (4 - 1 - math.log(4)) * latent_dim:.4f}][{0.5 * (4 - 1 - math.log(4)) * latent_dim:.4f}][{0.5 * (4 - 1 - math.log(4)) * latent_dim:.4f}]")

    _, _, kl_random = vae_loss(x=x, x_hat=x_hat, mean=mean, logvar=logvar, beta=beta)
    print(f" KL at random batch: min {kl_random.min():.4f}, expected: >=0")

    # reconstruction check
    _, perfect_recon, _ = vae_loss(x=x, x_hat=x, mean=mean, logvar=logvar, beta=beta)
    _, off_by_1_recon, _ = vae_loss(x=x, x_hat=x+1, mean=mean, logvar=logvar, beta=beta)
    print(f"[6] reconstruction check: perfect = {perfect_recon.abs().max()}, expected: 0.0; off by one = {off_by_1_recon.abs().max()}, expected: number of features = {n_features}")

    # per sample output and beta weightign check
    loss, reconstruction_loss, kl = vae_loss(x=x, x_hat=x_hat, mean=mean, logvar=logvar, beta=beta)
    print(f"[7] shapes: loss {tuple(loss.shape)}, reconstruction loss {tuple(reconstruction_loss.shape)}, KL divergence {tuple(kl.shape)}, expected (256,) on all")

    loss_b0, _, _ = vae_loss(x=x, x_hat=x_hat, mean=mean, logvar=logvar, beta=0)
    loss_b0_5, _, _ = vae_loss(x=x, x_hat=x_hat, mean=mean, logvar=logvar, beta=0.5)
    print(f" beta=0 -> loss==reconstruction loss: {torch.allclose(loss_b0, reconstruction_loss)}; beta=0.5 -> loss==reconstruction loss + 0.5*kl: {torch.allclose(loss_b0_5, reconstruction_loss + 0.5 * kl)} (expect True, True)")

    # gradient in enc and dec check
    vae.zero_grad()
    loss.mean().backward()
    missing = [n for n,p in vae.named_parameters() if p.grad is None]
    print(f"[8] Missing gradient check: {missing or 'none'}, expected: none")

    # behaviour with real data, if recon loss is unsually high, for example when seed=1, then we were just unlucky :(
    data = Data(conf=conf)
    X_s, y_s, _, feature_names = data.get_splits()
    train_loader, _, _ = data.make_dataloaders(X_s=X_s, y_s=y_s)
    (xb,) = next(iter(train_loader))
    vae_real = VAE(input_dim=len(feature_names), hidden_dims=conf.hidden_dims, latent_dim=conf.latent_dim, activation=conf.activation)
    with torch.no_grad():
        x_hat, mean, logvar = vae_real(x=xb)
        _, r, k = vae_loss(x=xb, x_hat=x_hat, mean=mean, logvar=logvar, beta=beta)

    print(f"[9] real batch results, untrained: reconstruction loss = {r.mean():.2f}, KL divergence = {k.mean():.3f}, expected: recon=~{len(feature_names)} and KL<1")