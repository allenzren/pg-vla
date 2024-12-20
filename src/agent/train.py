"""
Main training agent. Using bfloat16 and AMP.

"""

import logging
import os
import random
from collections import deque

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from src.agent.dataset import TorchRLDSInterleavedDataset
from src.model.vla.model import VLA
from src.model.vla.processing import VLAProcessor
from src.utils.metric import get_action_accuracy
from src.utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions

log = logging.getLogger(__name__)


class TrainAgent:
    def __init__(self, cfg):
        # seeding
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # devices
        self.gpu_id = cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.multi_gpu = cfg.multi_gpu
        world_size = 1  # single gpu
        if self.multi_gpu:
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            world_size = int(os.environ["WORLD_SIZE"])
            group_rank = int(os.environ["GROUP_RANK"])
            log.info(
                f"GPU local ID: {self.gpu_id}. Global rank: {global_rank}. Local rank: {local_rank}. Local world size: {local_world_size}. World size: {world_size}. Group rank: {group_rank}"
            )
            for i in range(torch.cuda.device_count()):
                log.info(
                    f"Local rank: {local_rank}, GPU UUID: {torch.cuda.get_device_properties(i).uuid}"
                )
        self.main_rank = not self.multi_gpu or global_rank == 0

        # training params
        self.n_updates = int(cfg.n_updates)
        self.max_grad_norm = cfg.max_grad_norm

        # model
        assert not (
            (cfg.quantize or cfg.lora) and not cfg.load_pretrained_weights
        ), "Please load pretrained weights if quantizing VLM or using Lora."
        if cfg.quantize and not cfg.lora:
            log.warning(
                "Quantizing VLM but not adding Lora weights, which means the VLM will be fully frozen!"
            )  # since the weights have requires_grad=False. However, we are not excluding the weights from the optimizer yet!
        self.model = VLA(cfg, use_ddp=self.multi_gpu)
        if cfg.resume_checkpoint_path:
            self.load_checkpoint(cfg.resume_checkpoint_path)
        elif cfg.load_pretrained_weights:
            self.model.load_pretrained_weights()
        self.model.freeze_unused_weights()
        if cfg.lora:
            self.model.freeze_non_lora_weights_in_vlm()
        self.model = self.model.to(torch.bfloat16)
        self.model.to(self.device)  # quantization happens
        log.info(f"Using cuda device: {self.device}")
        if self.multi_gpu:
            log.info(
                f"Using {local_world_size} GPUs in each of the {cfg.n_nodes} nodes"
            )
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                gradient_as_bucket_view=True,
                static_graph=False,
            )
            model = self.model.module
        else:
            model = self.model
        if self.multi_gpu:
            dist.barrier()
        log_allocated_gpu_memory(log, "loading model")

        # determine batch size and gradient accumulation steps
        self.grad_accumulation_steps = (
            cfg.global_batch_size // cfg.per_device_batch_size // world_size
        )

        # tokenizer --- assume paligemma for now
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )

        # processor
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )

        # dataloader --- use the same for all ranks, num_workers=0
        self.train_dataloader = DataLoader(
            TorchRLDSInterleavedDataset(cfg.data.train, train=True).dataset,
            batch_size=cfg.per_device_batch_size,
            pin_memory=True,
        )  # full bridge dataset has 2195527 transitions and 60064 trajectories
        self.run_eval = cfg.data.get("val")
        if self.run_eval:
            cfg_data_val = OmegaConf.merge(cfg.data.train, cfg.data.val)
            self.val_dataiterator = iter(
                DataLoader(
                    TorchRLDSInterleavedDataset(cfg_data_val, train=False).dataset,
                    batch_size=cfg.per_device_batch_size,
                    pin_memory=True,
                )
            )
            self.eval_thresholds = cfg.eval_thresholds
            self.eval_freq = cfg.eval_freq
            self.per_device_num_eval_batch = (
                cfg.eval_size // cfg.per_device_batch_size // world_size
            )
        log.info(f"Total number of gradient updates: {self.n_updates}")
        log.info(f"Global batch size: {cfg.global_batch_size}")
        log.info(f"Per device batch size: {cfg.per_device_batch_size}")
        log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")

        # optimizer - action only: 0.315B (0.333B with adaLN and time_dim=256),
        # rest: 2.359B (0.109B with lora rank 64, 0.055B with rank 32)
        self.train_vlm = cfg.train_vlm
        self.trained_parameters = model.action_expert_parameters
        if cfg.offload_optimizer:
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

            self.action_optimizer = CPUOffloadOptimizer(
                model.action_expert_parameters,
                torch.optim.AdamW,  # no need to use low-bit optimizer
                fused=True,
                offload_gradients=False,  # does not work with grad accumulation
                lr=cfg.action_lr,
                weight_decay=cfg.action_weight_decay,
            )
        else:
            import bitsandbytes as bnb

            self.action_optimizer = bnb.optim.AdamW8bit(
                model.action_expert_parameters,
                lr=cfg.action_lr,
                weight_decay=cfg.action_weight_decay,
            )
        self.action_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.action_optimizer,
            first_cycle_steps=cfg.action_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.action_lr,
            min_lr=cfg.action_lr_scheduler.min_lr,
            warmup_steps=cfg.action_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        log.info(
            f"Number of trained parameters (Action): {get_num_params_in_billions(self.action_optimizer):.3f}B"
        )
        if self.train_vlm:
            if cfg.lora:
                vlm_trained_parameters = model.lora_pretrained_parameters
            else:
                vlm_trained_parameters = model.pretrained_parameters
            self.trained_parameters += vlm_trained_parameters
            if cfg.offload_optimizer:
                self.vlm_optimizer = CPUOffloadOptimizer(
                    vlm_trained_parameters,
                    torch.optim.AdamW,
                    fused=True,
                    offload_gradients=False,
                    lr=cfg.vlm_lr,
                    weight_decay=cfg.vlm_weight_decay,
                )
            else:
                self.vlm_optimizer = bnb.optim.AdamW8bit(
                    vlm_trained_parameters,
                    lr=cfg.vlm_lr,
                    weight_decay=cfg.vlm_weight_decay,
                )
            self.vlm_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.vlm_optimizer,
                first_cycle_steps=cfg.vlm_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.vlm_lr,
                min_lr=cfg.vlm_lr_scheduler.min_lr,
                warmup_steps=cfg.vlm_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
            log.info(
                f"Number of trained parameters (VLM): {get_num_params_in_billions(self.vlm_optimizer):.3f}B"
            )
        if cfg.resume_checkpoint_path:
            self.load_optimizer(cfg.resume_checkpoint_path)

        # logging
        self.use_wandb = cfg.get("wandb", None)
        if self.use_wandb and self.main_rank:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                id=self.wandb_id if hasattr(self, "wandb_id") else None,
                resume="allow",  # not using resume_from
            )
        self.debug = cfg.get("debug", False)
        self.save_model_freq = int(cfg.save_model_freq)
        self.log_freq = cfg.log_freq
        self.log_dir = cfg.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run(self):
        timer = Timer()
        cnt_batch = 0 if not hasattr(self, "cnt_batch") else self.cnt_batch
        cnt_update = (
            0 if not hasattr(self, "cnt_update") else self.cnt_update
        )  # resume training if loaded checkpoint
        loss_train_deque = deque(maxlen=self.grad_accumulation_steps)
        new_eval_from_last_log = False
        if self.multi_gpu:
            import torch.distributed as dist
        self.model.train()

        def preprocess_batch(batch):
            images = batch["observation"]["image_primary"]
            proprios = batch["observation"]["proprio"]
            actions = batch["action"].squeeze(1)  # remove the time dimension
            texts = [
                text.decode("utf-8") for text in batch["task"]["language_instruction"]
            ]
            images = einops.rearrange(
                images, "B T H W C -> B (T C) H W"
            )  # remove cond_steps dimension
            model_inputs = self.processor(
                text=texts, images=images
            )  # TODO: move to dataset pre-processing
            return {
                "pixel_values": model_inputs["pixel_values"].to(self.device),
                "input_ids": model_inputs["input_ids"].to(self.device),
                "proprios": proprios.to(self.device),
                "actions": actions.to(self.device),
                "attention_mask": model_inputs["attention_mask"].to(
                    self.device
                ),  # with padding if bsz > 1
            }

        while 1:
            for batch in self.train_dataloader:
                """
                batch: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
                observation: 'image_primary' (torch.Size([bsz, 1, H, W, 3], uint8), 'image_wrist', 'timestep' (torch.Size([bsz, 1])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([bsz, window, 4]), 'proprio' (torch.Size([bsz, window, proprio_dim])
                task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([bsz]))
                action (torch.Size([bsz, window, horizon, action_dim], float32)
                action_pad_mask (torch.Size([bsz, window, horizon, action_dim]))
                """
                inputs = preprocess_batch(batch)
                if self.debug and cnt_batch == 0:
                    images = batch["observation"]["image_primary"]
                    proprios = batch["observation"]["proprio"]
                    actions = batch["action"].squeeze(1)
                    texts = [
                        text.decode("utf-8")
                        for text in batch["task"]["language_instruction"]
                    ]
                    log.info(f"{self.gpu_id} device {self.device}")
                    log.info(f"{self.gpu_id} texts {texts}")
                    log.info(f"{self.gpu_id} images {images.shape}")
                    log.info(
                        f"{self.gpu_id} actions {actions.shape} {actions.mean()} {actions.std()} {actions.max()} {actions.min()}"
                    )
                    log.info(
                        f"{self.gpu_id} proprios {proprios.shape} {proprios.mean()} {proprios.std()} {proprios.max()} {proprios.min()}"
                    )

                    # Save an image for debugging
                    image = images[0, 0].clone().to("cpu")
                    image = Image.fromarray(image.numpy())
                    image.save(os.path.join(self.log_dir, f"image_{self.gpu_id}.png"))

                # make sure only syncing when taking gradient steps
                if (cnt_batch + 1) % self.grad_accumulation_steps != 0:
                    with self.model.no_sync():
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                            loss_train = self.model(**inputs)
                        if self.debug:
                            log_allocated_gpu_memory(log, f"forward batch {cnt_batch}")
                        normalized_loss = loss_train / self.grad_accumulation_steps
                        normalized_loss.backward()
                        if self.debug:
                            log_allocated_gpu_memory(log, f"backward batch {cnt_batch}")
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        loss_train = self.model(**inputs)
                    if self.debug:
                        log_allocated_gpu_memory(log, f"forward batch {cnt_batch}")
                    normalized_loss = loss_train / self.grad_accumulation_steps
                    normalized_loss.backward()  # gradients synced
                    if self.debug:
                        log_allocated_gpu_memory(log, f"backward batch {cnt_batch}")

                    # step
                    torch.nn.utils.clip_grad_norm_(
                        self.trained_parameters,
                        max_norm=self.max_grad_norm,
                    )  # not work any more because of offload? no error thrown
                    self.action_optimizer.step()
                    self.action_lr_scheduler.step()
                    if self.train_vlm:
                        self.vlm_optimizer.step()
                        self.vlm_lr_scheduler.step()
                    if self.debug:
                        log_allocated_gpu_memory(
                            log, f"optimizer step batch {cnt_batch}"
                        )
                    self.action_optimizer.zero_grad(set_to_none=True)
                    if self.train_vlm:
                        self.vlm_optimizer.zero_grad(set_to_none=True)
                    if self.debug:
                        log_allocated_gpu_memory(
                            log, f"optimizer zero grad batch {cnt_batch}"
                        )
                    cnt_update += 1

                    # save model at the end of update, models just synced
                    if self.main_rank and (
                        cnt_update % self.save_model_freq == 0
                        or cnt_update == self.n_updates
                    ):
                        self.save_training(cnt_update, cnt_batch)

                # aggregate loss
                if self.multi_gpu:
                    dist.all_reduce(loss_train, op=dist.ReduceOp.SUM)
                    loss_train_deque.append(loss_train.item() / dist.get_world_size())
                else:
                    loss_train_deque.append(loss_train.item())

                # validation with action accuracy --- discretize actions (gripper treated as binary, otherwise 0.1 interval)
                if self.run_eval and (cnt_batch + 1) % self.eval_freq == 0:
                    new_eval_from_last_log = True
                    self.model.eval()
                    eval_accuracy = torch.zeros(len(self.eval_thresholds))
                    eval_l1_loss = torch.tensor(0.0)
                    log.info(
                        f"Running evaluation for {self.per_device_num_eval_batch} batches..."
                    )
                    with torch.no_grad():
                        for _ in range(self.per_device_num_eval_batch):
                            batch_eval = next(self.val_dataiterator)
                            inputs = preprocess_batch(batch_eval)
                            gt_actions = inputs.pop("actions")
                            with torch.autocast(
                                "cuda", dtype=torch.bfloat16, enabled=True
                            ):
                                if self.multi_gpu:
                                    preds = self.model.module.infer_action(**inputs)
                                else:
                                    preds = self.model.infer_action(**inputs)
                            eval_accuracy += get_action_accuracy(
                                gt_actions, preds, self.eval_thresholds
                            )
                            eval_l1_loss += torch.nn.functional.l1_loss(
                                preds, gt_actions
                            ).to("cpu")
                    eval_accuracy = eval_accuracy / self.per_device_num_eval_batch
                    eval_l1_loss = eval_l1_loss / self.per_device_num_eval_batch
                    self.model.train()
                    if self.multi_gpu:
                        dist.all_reduce(eval_accuracy, op=dist.ReduceOp.SUM)
                        dist.all_reduce(eval_l1_loss, op=dist.ReduceOp.SUM)
                        eval_accuracy /= dist.get_world_size()
                        eval_l1_loss /= dist.get_world_size()
                    log_msg = f"Eval | l1 Loss: {eval_l1_loss.item():.3f} | "
                    log_msg += " | ".join(
                        [
                            f"acc thres {threshold}: {accuracy.item():.3f}"
                            for threshold, accuracy in zip(
                                self.eval_thresholds, eval_accuracy, strict=False
                            )
                        ]
                    )
                    log.info(log_msg)

                # log loss
                if self.main_rank and cnt_batch % self.log_freq == 0:
                    loss_train_metric = np.mean(loss_train_deque)
                    log_msg = f"Batch {cnt_batch} Update {cnt_update}: t {timer():8.4f} | train loss {loss_train_metric:6.4f} | action lr {self.action_optimizer.param_groups[0]['lr']:10.8f}"
                    if self.train_vlm:
                        log_msg += f" | vlm lr {self.vlm_optimizer.param_groups[0]['lr']:10.8f}"
                    log.info(log_msg)
                    if self.use_wandb:
                        wandb_metrics = {
                            "loss - train": loss_train_metric,
                            "gradient steps": cnt_update,
                            "action lr": self.action_optimizer.param_groups[0]["lr"],
                        }
                        if self.train_vlm:
                            wandb_metrics["vlm lr"] = self.vlm_optimizer.param_groups[
                                0
                            ]["lr"]
                        if new_eval_from_last_log:
                            wandb_metrics.update(
                                {
                                    f"eval acc - thres {threshold}": accuracy.item()
                                    for threshold, accuracy in zip(
                                        self.eval_thresholds,
                                        eval_accuracy,
                                        strict=False,
                                    )
                                }
                            )
                            wandb_metrics["eval l1 loss"] = eval_l1_loss.item()
                            new_eval_from_last_log = False
                        wandb.log(wandb_metrics, step=cnt_batch, commit=True)

                # Count
                cnt_batch += 1
                if cnt_update >= self.n_updates:
                    return

    @log_execution_time()
    def save_training(self, cnt_update, cnt_batch):
        data = {
            "cnt_update": cnt_update,
            "cnt_batch": cnt_batch,
            "model": (
                self.model.module.state_dict()
                if self.multi_gpu
                else self.model.state_dict()
            ),
            "action_optimizer": self.action_optimizer.state_dict(),
            "vlm_optimizer": self.vlm_optimizer.state_dict()
            if self.train_vlm
            else None,
            "action_lr_scheduler": self.action_lr_scheduler.state_dict(),
            "vlm_lr_scheduler": self.vlm_lr_scheduler.state_dict()
            if self.train_vlm
            else None,
            "wandb_id": wandb.run.id if self.use_wandb else None,
        }
        savepath = os.path.join(self.checkpoint_dir, f"ckpt_{cnt_update}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    @log_execution_time()
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        self.cnt_update = data["cnt_update"]
        self.cnt_batch = data["cnt_batch"]
        self.wandb_id = data["wandb_id"]

        self.model.load_state_dict(data["model"], strict=True)
        log.info(
            f"Loaded model from {path} at update {self.cnt_update} batch {self.cnt_batch}"
        )

    @log_execution_time()
    def load_optimizer(self, path):
        """load to cpu first, then move to gpu"""
        from src.utils.optim import optimizer_to

        data = torch.load(path, weights_only=True, map_location="cpu")
        self.action_optimizer.load_state_dict(data["action_optimizer"])
        optimizer_to(self.action_optimizer, self.device)
        self.action_lr_scheduler.load_state_dict(data["action_lr_scheduler"])

        if self.train_vlm:
            self.vlm_optimizer.load_state_dict(data["vlm_optimizer"])
            optimizer_to(self.vlm_optimizer, self.device)
            self.vlm_lr_scheduler.load_state_dict(data["vlm_lr_scheduler"])
        log.info(f"Loaded optimizer and scheduler states from {path}")
