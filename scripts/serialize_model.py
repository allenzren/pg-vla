import random

import hydra
import numpy as np
import simpler_env
import torch
import torch_tensorrt
from omegaconf import OmegaConf

from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time


@log_execution_time()
def load_checkpoint(model, path):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(data["model"], strict=True)
    print(f"Loaded model from {path}")


def main(args):
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # devices
    device = torch.device(f"cuda:{args.gpu_id}")

    # load default configs
    if "fractal" in args.checkpoint_path:
        cfg = OmegaConf.load(
            "config/eval/fractal_apple.yaml"
        )  # doesn't matter which task
    if "bridge" in args.checkpoint_path:
        cfg = OmegaConf.load("config/eval/bridge.yaml")

    # determine flow matching schedule
    if "uniform" in args.checkpoint_path:
        cfg.flow_schedule = "uniform"
    if "gamma" in args.checkpoint_path:
        cfg.flow_schedule = "gamma"

    # model
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model = PiZeroInference(cfg, use_ddp=False)
    load_checkpoint(model, args.checkpoint_path)
    model.freeze_all_weights()
    model.to(dtype)
    model.to(device)
    model.eval()
    print(f"Using cuda device: {device} dtype: {dtype}")
    log_allocated_gpu_memory(None, "loading model", args.gpu_id)

    # simpler env
    env = simpler_env.make(args.task)

    # env specifics
    env_adapter = hydra.utils.instantiate(cfg.env.adapter)
    env_adapter.reset()

    # run an episode
    episode_id = random.randint(0, 20)
    env_reset_options = {}
    env_reset_options["obj_init_options"] = {
        "episode_id": episode_id,  # this determines the obj inits in bridge
    }
    obs, reset_info = env.reset(options=env_reset_options)
    instruction = env.get_language_instruction()
    print(
        f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
    )
    # infer action chunk
    inputs = env_adapter.preprocess(env, obs, instruction)
    causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
        model.build_causal_mask_and_position_ids(inputs["attention_mask"], dtype=dtype)
    )
    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
        causal_mask
    )
    inputs = [
        inputs["input_ids"],
        inputs["pixel_values"].to(dtype),
        image_text_proprio_mask,
        action_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
        inputs["proprios"].to(dtype),
    ]
    inputs = [i.to(device) for i in inputs]
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    print("compiling model...")
    trt_gm = torch_tensorrt.compile(model, inputs=inputs, truncate_long_and_double=True)
    torch_tensorrt.save(trt_gm, "trt.ep", inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=inputs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="google_robot_pick_horizontal_coke_can",
        choices=[
            "widowx_carrot_on_plate",
            "widowx_put_eggplant_in_basket",
            "widowx_spoon_on_towel",
            "widowx_stack_cube",
            "google_robot_pick_horizontal_coke_can",
            "google_robot_pick_vertical_coke_can",
            "google_robot_pick_standing_coke_can",
            "google_robot_move_near_v0",
            "google_robot_open_drawer",
            "google_robot_close_drawer",
            "google_robot_place_apple_in_closed_top_drawer",
        ],
    )
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true")
    args = parser.parse_args()

    # check task
    if "google_robot" in args.task:
        assert "fractal" in args.checkpoint_path
    if "widowx" in args.task:
        assert "bridge" in args.checkpoint_path

    main(args)
