import functools
import os

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from giga_models import utils as gm_utils
from world_action_model.models import CasualWorldActionTransformer_MoT, rtc as rtc_lib
from giga_train import ModuleDict, Trainer


def _diffsynth_wan_to_diffusers(state_dict):
    """Convert a DiffSynth Wan DiT checkpoint to the names used by Giga/diffusers."""
    replacements = (
        (".self_attn.norm_k.", ".attn1.norm_k."),
        (".self_attn.norm_q.", ".attn1.norm_q."),
        (".self_attn.k.", ".attn1.to_k."),
        (".self_attn.q.", ".attn1.to_q."),
        (".self_attn.v.", ".attn1.to_v."),
        (".self_attn.o.", ".attn1.to_out.0."),
        (".cross_attn.norm_k_img.", ".attn2.norm_added_k."),
        (".cross_attn.k_img.", ".attn2.add_k_proj."),
        (".cross_attn.v_img.", ".attn2.add_v_proj."),
        (".cross_attn.norm_k.", ".attn2.norm_k."),
        (".cross_attn.norm_q.", ".attn2.norm_q."),
        (".cross_attn.k.", ".attn2.to_k."),
        (".cross_attn.q.", ".attn2.to_q."),
        (".cross_attn.v.", ".attn2.to_v."),
        (".cross_attn.o.", ".attn2.to_out.0."),
        (".ffn.0.", ".ffn.net.0.proj."),
        (".ffn.2.", ".ffn.net.2."),
        (".norm3.", ".norm2."),
    )
    top_level = {
        "text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
        "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
        "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
        "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
        "time_projection.1.": "condition_embedder.time_proj.",
        "img_emb.proj.0.": "condition_embedder.image_embedder.norm1.",
        "img_emb.proj.1.": "condition_embedder.image_embedder.ff.net.0.proj.",
        "img_emb.proj.3.": "condition_embedder.image_embedder.ff.net.2.",
        "img_emb.proj.4.": "condition_embedder.image_embedder.norm2.",
        "head.modulation": "scale_shift_table",
        "head.head.": "proj_out.",
    }
    converted = {}
    for source_name, tensor in state_dict.items():
        name = source_name
        if name.startswith("blocks."):
            if name.endswith(".modulation"):
                name = name.removesuffix(".modulation") + ".scale_shift_table"
            else:
                for source, target in replacements:
                    if source in name:
                        name = name.replace(source, target)
                        break
        else:
            for source, target in top_level.items():
                if name == source or name.startswith(source):
                    name = target + name[len(source):]
                    break
        converted[name] = tensor
    return converted


class CasualWATrainerMoT(Trainer):
    """WAM pretraining trainer for the Mixture-of-Transformers policy."""

    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        self.visual_flow_shift = float(model_config.visual_flow_shift)
        self.action_flow_shift = float(model_config.action_flow_shift)
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.action_repeats = model_config.get("action_repeats", 1)
        self.state_repeats = model_config.get("state_repeats", 1)
        self.action_dim = int(model_config.get("action_dim", 14))
        self.num_embodiments = int(model_config.get("num_embodiments", 1))

        self.rtc = rtc_lib.RTCConfig.from_dict(model_config.get("rtc", None))
        self._rtc_validated = False
        if self.rtc.enabled and self.action_repeats != 1:
            raise ValueError(
                "rtc.enabled=true requires action_repeats == 1; "
                "the RTC prefix mask assumes one token per control step"
            )
        if self.is_main_process:
            self.logger.info(self.rtc.describe())

        vae_pretrained = model_config.get("vae_pretrained", os.path.join(pretrained, "vae"))
        vae_dtype = model_config.get("vae_dtype", torch.float32)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        transformer_pretrained = model_config.get("transformer_pretrained", os.path.join(pretrained, "transformer"))
        transformer_cfg = model_config.get("transformer")
        transformer = CasualWorldActionTransformer_MoT(**transformer_cfg)
        transformer = load_pretrained_weights(
            transformer,
            transformer_pretrained,
            skip_action_expert=model_config.get("skip_action_expert", False),
            strict_load=model_config.get("strict_load", False),
            require_full_load=model_config.get("require_full_load", False),
        )
        video_dit_pretrained = model_config.get("video_dit_pretrained", None)
        if video_dit_pretrained:
            from safetensors.torch import load_file

            video_state = _diffsynth_wan_to_diffusers(load_file(video_dit_pretrained))
            loaded, skipped, unexpected, _ = transformer.load_from_wan_pretrained_state_dict(
                video_state, skip_action_expert=True, verbose=True
            )
            if len(loaded) < 800 or unexpected:
                raise RuntimeError(
                    f"Invalid Video DiT override: loaded={len(loaded)}, "
                    f"skipped={len(skipped)}, unexpected={len(unexpected)}"
                )
            print(
                f"Loaded DiffSynth Video DiT from {video_dit_pretrained}: "
                f"loaded={len(loaded)}, skipped={len(skipped)}, unexpected=0"
            )
            del video_state
        transformer.to(self.device)
        if model_config.get("enable_gradient_checkpointing", True):
            transformer.enable_gradient_checkpointing()

        model = dict(transformer=transformer)
        checkpoint = model_config.get("checkpoint", None)
        strict = model_config.get("strict", True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()
        return model

    def forward_step(self, batch_dict):
        transformer = functools.partial(self.model, "transformer")
        images = batch_dict["images"]
        bs = images.shape[0]
        prompt_embeds = batch_dict["prompt_embeds"]
        action = batch_dict["action"]
        state = batch_dict["state"]
        embodiment_id = batch_dict["embodiment_id"]

        visual_timestep, visual_sigma = self.get_timestep_and_sigma(bs, images.ndim, self.visual_flow_shift)
        action_timestep, action_sigma = self.get_timestep_and_sigma(bs, action.ndim, self.action_flow_shift)

        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)

        visual_latents = self.forward_vae(images)
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * visual_sigma + visual_latents * (1 - visual_sigma)

        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        prefix_mask = None
        num_action_tokens = action.shape[1]
        if self.rtc.enabled:
            if not self._rtc_validated:
                rtc_lib.validate(self.rtc, action_horizon=num_action_tokens)
                self._rtc_validated = True
                if self.is_main_process:
                    self.logger.info(
                        f"RTC validated for action_horizon={num_action_tokens}"
                    )
            delay = rtc_lib.sample_delay(bs, self.rtc, action.device)
            prefix_mask = rtc_lib.build_prefix_mask(delay, num_action_tokens)
            action_sigma = rtc_lib.apply_prefix_sigma(
                action_sigma.reshape(bs), prefix_mask
            )
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)
        if self.rtc.enabled:
            noisy_action = rtc_lib.jitter_prefix(
                noisy_action, prefix_mask, self.rtc.prefix_jitter_std
            )

        prompt_embeds = prompt_embeds.to(self.dtype)
        if "ref_images" not in batch_dict:
            raise ValueError("CasualWATrainerMoT requires ref_images in batch_dict")

        if not self.expand_timesteps:
            ref_images = batch_dict["ref_images"]
            ref_latents = self.forward_vae(ref_images)
            num_frames = images.shape[1]
            batch_size = ref_latents.shape[0]
            latent_height = ref_latents.shape[-2]
            latent_width = ref_latents.shape[-1]
            mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
            mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
            mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
            mask_lat_size = mask_lat_size.transpose(1, 2).to(ref_latents.device)
            condition = torch.concat([mask_lat_size, ref_latents], dim=1)
            insert_noisy_latents = torch.concat([noisy_latents, condition], dim=1)
        else:
            num_latent_frames = visual_latents.shape[2]
            latent_height = visual_latents.shape[-2]
            latent_width = visual_latents.shape[-1]
            ref_images = batch_dict["ref_images"][:, :1]
            ref_latents = self.forward_vae(ref_images)
            first_frame_mask = torch.ones(
                bs,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=visual_latents.dtype,
                device=visual_latents.device,
            )
            first_frame_mask[:, :, 0] = 0
            insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents

        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        num_state_tokens = state.shape[1]
        noisy_action = noisy_action.to(self.dtype)
        state = state.to(self.dtype)
        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        num_clean_latent_tokens = frame_per_tokens
        timestep = torch.zeros(
            bs,
            num_state_tokens + num_action_tokens + num_latent_tokens,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        action_t_start = num_state_tokens + num_clean_latent_tokens
        if self.rtc.enabled:
            timestep[:, action_t_start : action_t_start + num_action_tokens] = (
                rtc_lib.apply_prefix_timestep(
                    action_timestep.to(timestep.dtype), prefix_mask
                )
            )
        else:
            timestep[:, action_t_start : action_t_start + num_action_tokens] = (
                action_timestep[:, None]
            )
        timestep[:, num_state_tokens + num_action_tokens + num_clean_latent_tokens :] = visual_timestep[:, None]

        visual_pred, action_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            embodiment_id=embodiment_id,
        )

        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()

        action_loss = rtc_lib.masked_action_loss(
            action_pred, action_target, prefix_mask
        )
        losses = {
            "visual_loss": visual_loss,
            "action_loss": action_loss,
        }
        if self.rtc.enabled:
            losses["rtc_mean_delay"] = delay.float().mean()
            losses["rtc_postfix_frac"] = (~prefix_mask).float().mean()
        return losses

    def parse_losses(self, losses):
        """Log RTC diagnostics without adding them to the optimization loss."""
        metric_keys = [key for key in losses if key.startswith("rtc_")]
        metrics = {key: losses.pop(key) for key in metric_keys}
        loss = super().parse_losses(losses)
        for key, value in metrics.items():
            value = self.accelerator.gather(value.detach()).float().mean()
            if key not in self._outputs:
                self._outputs[key] = {"sum": 0.0, "num": 0}
            self._outputs[key]["sum"] += value.item()
            self._outputs[key]["num"] += 1
        return loss

    def set_ema_models(self):
        """Allow RTC fine-tuning to use a shorter EMA window."""
        decay = float(self.kwargs.get("ema_decay", 0.9999))
        if decay == 0.9999:
            return super().set_ema_models()

        from giga_train.strategies.ema import EMAModel

        if self.is_main_process:
            self.logger.info(f"EMA decay overridden to {decay}")
        for model in self.models:
            ema_model = EMAModel(
                decay=decay,
                rank=self.process_index,
                world_size=self.num_processes,
            )
            ema_model.load_state_dict(model.state_dict(), device=self.device)
            self.ema_models.append(ema_model)

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, "b t c h w -> b c t h w")
            latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents

    def get_timestep_and_sigma(self, batch_size, ndim, flow_shift):
        sigma = torch.rand(batch_size).to(self.device)
        sigma = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return timestep, sigma

def load_pretrained_weights(
    model,
    pretrained_path,
    skip_action_expert=False,
    strict_load=False,
    require_full_load=False,
):
    from safetensors.torch import load_file

    pretrained_path = gm_utils.get_model_path(pretrained_path)
    if os.path.isdir(pretrained_path):
        weight_files = sorted(
            os.path.join(pretrained_path, f)
            for f in os.listdir(pretrained_path)
            if f.endswith((".safetensors", ".bin"))
        )
    else:
        weight_files = [pretrained_path]

    state_dicts = {}
    for weight_file in weight_files:
        print(f"Loading weights from {weight_file}")
        if weight_file.endswith(".bin"):
            checkpoint = torch.load(weight_file, map_location="cpu")
        else:
            checkpoint = load_file(weight_file)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        state_dicts.update(checkpoint)

    if not strict_load and hasattr(model, "load_from_wan_pretrained_state_dict"):
        if require_full_load:
            raise ValueError(
                "require_full_load=true requires strict_load=true; "
                "Wan remapping intentionally skips checkpoint keys"
            )
        loaded, skipped, unexpected, missing = model.load_from_wan_pretrained_state_dict(
            state_dicts,
            skip_action_expert=skip_action_expert,
        )
        if unexpected:
            print(f"[WARNING] Unexpected keys in Wan checkpoint: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
        if skipped:
            print(f"[WARNING] Skipped keys during MoT remap: {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
        if missing:
            print(f"[WARNING] Missing keys after MoT remap: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print(f"Loaded {len(loaded)} tensors from Wan2.2 into MoT.")
        return model

    missing_keys, unexpected_keys = model.load_state_dict(state_dicts, strict=False)
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys in state_dict: {unexpected_keys}")
    if missing_keys:
        print(f"[WARNING] Missing keys in state_dict: {missing_keys}")
    if require_full_load and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Incomplete warm start from {pretrained_path}: "
            f"{len(missing_keys)} missing / {len(unexpected_keys)} unexpected; "
            f"missing[:10]={list(missing_keys)[:10]}, "
            f"unexpected[:10]={list(unexpected_keys)[:10]}"
        )
    if require_full_load:
        print(
            f"Warm start verified: all {len(state_dicts)} tensors matched "
            "(0 missing / 0 unexpected)."
        )
    return model
