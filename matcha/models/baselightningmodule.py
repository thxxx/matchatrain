"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from matcha import utils
from matcha.utils.utils import plot_tensor
from matcha.utils.metric import wer_en, cer_en

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from contextlib import contextmanager

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        # 여기가 진짜 infrence 코드인데 모델 코드 자체에서 forward에 loss도 계산한다?
        dur_loss, prior_loss, diff_loss, *_ = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
            out_size=self.out_size,
            durations=batch["durations"],
        )
        
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def get_metric(self, vocoder, denoiser):
        """
        임의로 1000개 정도 생성한 뒤 wer, cer 체크
        """
        import pandas as pd
        import torchaudio
        from matcha.text import sequence_to_text, text_to_sequence
        from matcha.utils.utils import intersperse

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3-turbo"

        wmodel = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).eval()
        wmodel.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=wmodel,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        @torch.inference_mode()
        def to_waveform(mel, vocoder):
            audio = vocoder(mel).clamp(-1, 1)
            audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
            return audio.cpu().squeeze()
        
        n_timesteps = 10
        length_scale=1.0
        temperature = 0.667

        @torch.inference_mode()
        def process_text(text: str):
            x = torch.tensor(intersperse(text_to_sequence(text, ['english_cleaners2'])[0], 0),dtype=torch.long, device=device)[None]
            x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
            x_phones = sequence_to_text(x.squeeze(0).tolist())
            return {
                'x_orig': text, # Hi how are you today?
                'x': x,         # ids of phoneme embedding
                'x_lengths': x_lengths,
                'x_phones': x_phones # _h_ˈ_a_ɪ_ _h_ˌ_a_ʊ_ _ɑ_ː_ɹ_ _j_u_ː_ _t_ə_d_ˈ_e_ɪ_?_
            }
        
        @torch.inference_mode()
        def synthesise(text, spks=None):
            text_processed = process_text(text)
            
            output = self.synthesise(
                text_processed['x'], 
                text_processed['x_lengths'],
                n_timesteps=n_timesteps,
                temperature=temperature,
                spks=spks,
                length_scale=length_scale
            )
            # merge everything to one dict    
            output.update({**text_processed})
            return output

        eval_df = pd.read_csv("/workspace/matchatrain/LJSpeech-1.1/test.csv", sep="|", header=None)
        texts =  [eval_df.iloc[i][1] for i in range(len(eval_df))]
        resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)

        wer_list, cer_list = [], []

        for text in texts:
            with torch.no_grad():
                output = synthesise(text, spks=None)
                waveform = to_waveform(output['mel'], vocoder)
                mono_22k = resampler(waveform)
                script = pipe(mono_22k)

            _w = wer_en(text, script['text'])
            _c = cer_en(text, script['text'])
            wer_list.append(_w)
            cer_list.append(_c)

        # self.log("metric/WER", sum(wer_list)/len(wer_list), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        # self.log("metric/CER", sum(cer_list)/len(cer_list), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        wer = sum(wer_list)/len(wer_list)
        cer = sum(cer_list)/len(cer_list)

        step = self.current_epoch
        if hasattr(self.logger, "experiment"):
            exp = self.logger.experiment
            # TensorBoard 예시
            if hasattr(exp, "add_scalar"):
                exp.add_scalar("metric/WER", wer, step)
                exp.add_scalar("metric/CER", cer, step)
            # WandB 예시
            elif hasattr(exp, "log"):
                exp.log({"metric/WER": wer, "metric/CER": cer, "epoch": step})

        del wmodel, processor, pipe, resampler, eval_df, texts, wer_list, cer_list
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self) -> None:
        # 검증 전체를 EMA 가중치로 수행
        self._ema_ctx = self._swap_to_ema_weights()
        self._ema_ctx.__enter__()

    def on_validation_end(self) -> None:
        """
        validation이 다 끝난 뒤 한번 호출된다. = 현재는 1에포크당 1번
        """
        
        def load_vocoder(checkpoint_path):
            h = AttrDict(v1)
            hifigan = HiFiGAN(h).to(self.device)
            hifigan.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['generator'])
            _ = hifigan.eval()
            hifigan.remove_weight_norm()
            return hifigan
        
        vocoder = load_vocoder('/workspace/generator_v1')
        denoiser = Denoiser(vocoder, mode='zeros')
        
        @torch.inference_mode()
        def to_waveform(mel, vocoder):
            audio = vocoder(mel).clamp(-1, 1)
            audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
            return audio.cpu().squeeze()

        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                
                with torch.no_grad():
                    output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks)
                
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]

                waveform = to_waveform(output['mel'], vocoder)
                
                attn = output["attn"]
                
                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                if waveform is not None:
                    if waveform.dim() == 2:        # (B, T) -> (1, T)
                        waveform = waveform[:1]
                    if waveform.dim() == 3:        # (B, C, T) -> mono만
                        waveform = waveform[:1, :1, :]  # (1, 1, T)
    
                    # TensorBoard에 오디오 로깅
                    self.logger.experiment.add_audio(
                        f"audio/{i}",
                        waveform.squeeze().detach().cpu(),  # (C, T) 혹은 (T,)
                        global_step=self.current_epoch,
                        sample_rate=22050,
                    )
        
        if self.current_epoch % 5 == 4:
            self.get_metric(vocoder, denoiser)
        
        del vocoder
        del denoiser
        torch.cuda.empty_cache()

        if hasattr(self, "_ema_ctx"):
            self._ema_ctx.__exit__(None, None, None)
            del self._ema_ctx
        
        # ===== EMA utilities =====
    def ema_enabled(self) -> bool:
        # 필요시 하이퍼파라미터/Config로 제어 가능
        return getattr(self, "use_ema", True)

    def _ema_decay(self) -> float:
        # 보편적으로 0.999~0.9999, diffusion은 0.999~0.9995 많이 씀
        return float(getattr(self, "ema_decay", 0.999))

    def _ema_every(self) -> int:
        # 몇 스텝에 한 번 업데이트할지 (기본 1: 매 스텝)
        return int(getattr(self, "ema_every_n_steps", 1))

    def _ema_device(self):
        # EMA를 CPU에 둘 수도 있음(메모리 절약). 기본은 모델과 동일 디바이스.
        return getattr(self, "ema_device", None)  # None이면 param.device

    def _should_init_ema(self) -> bool:
        return not hasattr(self, "_ema_state_dict")

    def _init_ema_state(self):
        # 파라미터+버퍼를 통째로 복사 (버퍼는 decay 없이 그대로 보관)
        self._ema_state_dict = {}
        for k, v in self.state_dict().items():
            if torch.is_tensor(v):
                dev = self._ema_device() or v.device
                self._ema_state_dict[k] = v.detach().clone().to(dev)
            else:
                self._ema_state_dict[k] = v

    def _update_ema(self):
        if not self.ema_enabled():
            return
        if self._should_init_ema():
            self._init_ema_state()

        if self.global_step % self._ema_every() != 0:
            return

        d = self._ema_decay()
        with torch.no_grad():
            curr = self.state_dict()
            for k, v in curr.items():
                if not torch.is_tensor(v):
                    # non-tensor 그대로 유지
                    self._ema_state_dict[k] = v
                    continue
                ema_v = self._ema_state_dict[k]
                # 디바이스 맞추기 (EMA를 CPU에 두는 경우도 고려)
                if ema_v.device != (self._ema_device() or v.device):
                    ema_v = ema_v.to(self._ema_device() or v.device)
                    self._ema_state_dict[k] = ema_v
                # dtype mismatch 시 맞춤
                if ema_v.dtype != v.dtype:
                    ema_v = ema_v.to(dtype=v.dtype)
                    self._ema_state_dict[k] = ema_v
                # EMA 업데이트
                ema_v.mul_(d).add_(v.detach(), alpha=(1.0 - d))

    @contextmanager
    def _swap_to_ema_weights(self):
        """
        with self._swap_to_ema_weights():  # EMA 가중치로 검증/샘플링
            ...
        블록이 끝나면 원래 가중치로 복원
        """
        if not self.ema_enabled():
            yield
            return
        if self._should_init_ema():
            # 아직 학습 초기 등으로 EMA가 없다면 현재 가중치로 초기화
            self._init_ema_state()

        # 백업 -> EMA 로드 -> 작업 -> 복원
        backup = {k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in self.state_dict().items()}

        # EMA state를 현재 디바이스/dtype에 맞춰 로드
        load_sd = {}
        for k, v in self._ema_state_dict.items():
            if torch.is_tensor(v):
                dev = backup[k].device  # 현재 모델 텐서의 디바이스/dtype 기준
                load_sd[k] = v.to(device=dev, dtype=backup[k].dtype)
            else:
                load_sd[k] = v
        self.load_state_dict(load_sd, strict=True)
        try:
            yield
        finally:
            self.load_state_dict(backup, strict=True)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # 1) 원래 step 수행
        optimizer.step(closure=optimizer_closure)

        # 2) EMA 업데이트
        if self.ema_enabled():
            self._update_ema()

        # 3) 스케줄러 등은 Lightning이 알아서 호출 (configure_optimizers 반환 방식에 따라 다름)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema_enabled() and not self._should_init_ema():
            # CPU로 저장하면 안전
            ema_cpu = {}
            for k, v in self._ema_state_dict.items():
                if torch.is_tensor(v):
                    ema_cpu[k] = v.detach().cpu()
                else:
                    ema_cpu[k] = v
            checkpoint["ema_state_dict"] = ema_cpu
            checkpoint["ema_decay"] = self._ema_decay()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 기존 코드 유지
        self.ckpt_loaded_epoch = checkpoint["epoch"]
        # EMA 복구
        if "ema_state_dict" in checkpoint:
            self._ema_state_dict = {}
            for k, v in checkpoint["ema_state_dict"].items():
                self._ema_state_dict[k] = v  # 텐서는 나중에 swap시 디바이스/dtype 맞춰짐
            if "ema_decay" in checkpoint:
                self.ema_decay = float(checkpoint["ema_decay"])


    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})

    def sample_with_ema(self, *args, **kwargs):
        with self._swap_to_ema_weights():
            return self.synthesise(*args, **kwargs)
