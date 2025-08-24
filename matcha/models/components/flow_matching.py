from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger
import random
import numpy as np

log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.Ke_coef = 2

        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def get_timestep(self, batch_size):
        if random.random()<0.25:
            t = torch.rand((batch_size, ), dtype=self.dtype, device=self.device)
        else:
            tnorm = np.random.normal(loc=0, scale=1.0, size=batch_size)
            t = 1 / (1 + np.exp(-tnorm))
            t = torch.tensor(t, dtype=self.dtype, device=self.device)
        # t = torch.rand([batch_size, 1, 1], device=self.device, dtype=self.dtype)
        return t

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """
        여기서 train step의 flow matching loss 계산!
        
        Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b = x1.shape[0]
        K = int(self.Ke_coef)  # K >= 1

        # --- 1) 배치 확장(복제) 준비 ---
        # K==1이면 기존 경로 그대로 갑니다.
        if K > 1:
            # x1, mu, mask, spks(있다면)만 "복제" -> encoder/준비 연산을 재사용
            x1_rep   = x1.repeat_interleave(K, dim=0)
            mu_rep   = mu.repeat_interleave(K, dim=0)
            mask_rep = mask.repeat_interleave(K, dim=0)
            spks_rep = spks.repeat_interleave(K, dim=0) if spks is not None else None
        else:
            x1_rep, mu_rep, mask_rep, spks_rep = x1, mu, mask, spks

        K_batch = b * K  # 내부적으로 확장된 배치 크기

        # --- 2) timestep만 K배 샘플링 ---
        # get_timestep(K_batch): [K_batch] 또는 [K_batch, 1, ...] 형태여야 함
        t = self.get_timestep(K_batch)  # 0~1 uniform

        # x1_rep과 브로드캐스트 되도록 view로 맞춰줍니다.
        # (예: x1_rep: [K_batch, C, T]라면 t -> [K_batch, 1, 1])
        t = t.view(K_batch, *([1] * (x1_rep.dim() - 1)))

        # --- 3) 노이즈는 "복제"가 아니라 각 샘플마다 새로 샘플링 ---
        # (편향 없이 gradient를 얻기 위해 중요)
        z = torch.randn_like(x1_rep)

        # --- 4) forward 공식을 그대로 확장 배치에 적용 ---
        x_t = (1 - (1 - self.sigma_min) * t) * z + t * x1_rep
        target_vf = x1_rep - (1 - self.sigma_min) * z

        # estimator는 확장된 배치로 한 번 호출 (내부는 당연히 K배로 병렬 처리)
        # t 입력이 1D를 요구한다면 squeeze 필요. (대부분 [K_batch] 또는 [K_batch,1] 사용)
        t_for_net = t.squeeze()
        predicted_flow = self.estimator(x_t, mask_rep, mu_rep, t_for_net, spks_rep)

        # --- 5) 손실 정규화 ---
        # mask도 K배로 복제되었으므로 sum(mask) 역시 K배가 되어 자연스럽게 정규화가 유지됩니다.
        denom = (mask_rep.sum() * predicted_flow.shape[1]).clamp_min(1).to(x1_rep.dtype)
        loss = F.mse_loss(predicted_flow, target_vf, reduction="sum") / denom

        return loss, x_t


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
