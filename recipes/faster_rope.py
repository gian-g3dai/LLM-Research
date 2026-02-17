import torch
import time
from typing import Tuple, Optional


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device=None):
  # compute for each position the intensity of the embedding rotation
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # operation on complex numbers
  return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(
  xq: torch.Tensor,
  xk: torch.Tensor,
  freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  # convert float to complex numbers by joining pairs of dimensions together
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  # apply rotation (the multplication) and convert back to float
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis_new(
  dim: int,
  end: int,
  theta: float = 10000.0,
  device: Optional[torch.device] = None,
) -> torch.Tensor:
  device = device or torch.device("cpu")
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)
                           [: (dim // 2)].float() / dim))
  t = torch.arange(end, device=device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_sin = torch.sin(freqs)
  freqs_cos = torch.cos(freqs)
  freqs_cis = torch.stack([freqs_sin, freqs_cos], dim=-1)
  return freqs_cis

def reshape_for_broadcast_new(freqs_cis: torch.Tensor, x: torch.Tensor):
  ndim = x.ndim
  # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(1, x.shape[1], 1, x.shape[-1])

def _rotate(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
  x_real = x[..., 0::2]
  x_imag = x[..., 1::2]

  freq_cos = reshape_for_broadcast_new(cos, x_real).float()
  freq_sin = reshape_for_broadcast_new(sin, x_imag).float()

  x_out_real = x_real * freq_cos - x_imag * freq_sin
  x_out_imag = x_real * freq_sin + x_imag * freq_cos
  x_out = torch.stack([x_out_real, x_out_imag], dim=-1).flatten(-2)
  return x_out.type_as(x)


# 0.00025063300132751465
# @torch.jit.script
def cos_sin_apply_rotary_emb_new(
  xq: torch.Tensor,
  xk: torch.Tensor,
  freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  sin, cos = freqs_cis[..., 0], freqs_cis[..., 1]
  return _rotate(xq, sin, cos), _rotate(xk, sin, cos)


def precompute_freqs_cis_ted(
    seq_len: int, n_elem: int, base: int = 10000, device = None
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb_ted(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def main():
  num_iters = 1000
  device = torch.device("cuda")

  rope_times = []

  bsz = 16
  seqlen = 1
  n_local_heads = 40
  head_dim = 128


  for _ in range(num_iters):
    xq = torch.randn(bsz, seqlen, n_local_heads, head_dim, device=device) # bsz, seqlen, self.n_local_heads, self.head_dim
    xk = torch.randn(bsz, seqlen, n_local_heads, head_dim, device=device)

    freqs_cis = precompute_freqs_cis_new(
      dim = 128,
      end = 1,
      theta = 10_000,
      device = device
    )
    
    # freqs_cis = precompute_freqs_cis_ted(
    #   seq_len = seqlen,
    #   n_elem = head_dim,
    #   base = 10_000,
    #   device = device
    # )

    # freq_cis_og = precompute_freqs_cis(
    #   dim = 128,
    #   end = 1,
    #   theta = 10_000,
    #   device = device
    # )

    time_start = time.time()
    
    # xq_rot_og, xk_rot_og = apply_rotary_emb(xq, xk, freq_cis_og)

    xq_rot, xk_rot = cos_sin_apply_rotary_emb_new(xq, xk, freqs_cis)
    
    
    # xq_rot = apply_rotary_emb_ted(xq, freqs_cis)
    # xk_rot = apply_rotary_emb_ted(xk, freqs_cis)

    # print(torch.allclose(xq_rot, xq_rot_og))
    # print(torch.allclose(xk_rot, xk_rot_og))

    # break

    torch.cuda.synchronize()

    time_end = time.time()
    print("Time taken: ", time_end - time_start)
    rope_times.append(time_end - time_start)

  print("Average time taken: ", sum(rope_times) / len(rope_times))


if __name__ == "__main__":
  main()

# og



# new
# Average time taken:  0.00013414478302001953


# ted
# Average time taken:  0.00015071535110473632