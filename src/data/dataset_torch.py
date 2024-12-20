"""
From: https://github.com/octo-models/octo/blob/main/examples/06_pytorch_oxe_dataloader.py

This example shows how to use the `src.data` dataloader with PyTorch by wrapping it in a simple PyTorch dataloader. The config below also happens to be our exact pretraining config (except for the batch size and shuffle buffer size, which are reduced for demonstration purposes).
"""

import time

import tensorflow as tf
import torch
import tqdm
from torch.utils.data import DataLoader

from src.data.dataset import make_interleaved_dataset
from src.data.oxe import make_oxe_dataset_kwargs_and_weights

tf.config.set_visible_devices([], "GPU")


class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        # TODO: account for sample weights?
        return self._rlds_dataset.true_total_length
        # lengths = np.array(
        #     [
        #         stats["num_transitions"]
        #         for stats in self._rlds_dataset.dataset_statistics.values()
        #     ],
        #     dtype=float,
        # )
        # if hasattr(self._rlds_dataset, "sample_weights"):
        #     lengths *= self._rlds_dataset.sample_weights
        # total_len = lengths.sum()
        # if self._is_train:
        #     return int(0.95 * total_len)
        # else:
        #     return int(0.05 * total_len)


if __name__ == "__main__":
    import argparse

    import einops

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/n/fs/llm-unc/data/resize_224"
    )
    parser.add_argument("--mix", type=str, default="bridge")
    parser.add_argument("--camera_views", nargs="*", default=("primary",))
    parser.add_argument("--load_proprio", action="store_true")
    args = parser.parse_args()

    # config
    start_time = time.time()
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        args.mix,
        args.data_path,
        load_depth=False,
        load_language=True,
        load_proprio=args.load_proprio,
        load_camera_views=args.camera_views,
    )

    # dataset
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=True,
        split="train[:95%]",
        shuffle_buffer_size=10000,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
        batch_size=None,  # batching will be handles in PyTorch Dataloader object
        balance_weights=True,
        traj_transform_kwargs=dict(  # no neeed for goal relabeling
            window_size=2,
            action_horizon=4,
            subsample_length=100,
            skip_unlabeled=True,  # skip ones without language annotation
            # max_action_from_stats=True,
            # max_proprio_from_stats=True,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(
                        scale=[0.8, 1.0],
                        ratio=[0.9, 1.1],
                    ),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
                "wrist": dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            },
            resize_size=dict(
                primary=(224, 224),
                wrist=(224, 224),
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,
        traj_read_threads=48,
    )

    # convert for torch
    pytorch_dataset = TorchRLDSDataset(dataset)
    print("Dataset length (traj):", len(pytorch_dataset))
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=16,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )
    prep_time = time.time()
    print(f"Preparation time: {prep_time - start_time:.2f}s")

    print("Starting dataloader")
    cnt_batch = 0
    for _, _sample in tqdm.tqdm(enumerate(dataloader)):
        # _sample: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
        # observation: 'image_primary' (torch.Size([16, 2, 256, 256, 3]), 'image_wrist', 'timestep' (torch.Size([16, 2])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([16, 2, 4]), 'proprio' (fractal: torch.Size([16, 2, 8]))
        # task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([16]))
        # action (torch.Size([16, 2, 4, 7])
        # dataset_name
        # action_pad_mask (torch.Size([16, 2, 4, 7]))

        # timestep_pad_mask: which observations at the beginning of the trajectory are padding --- repeat the first observation at the beginning of the trajectory rather than going out of bounds
        # action_pad_mask: mark actions past the goal timestep as padding --- repeat the last action at the end of the trajectory rather than going out of bounds
        # task_completed should correspond to action_pad_mask
        # timestep should correspond to timestep_pad_mask (e.g., timestep [0, 0] for a datapoint indicates padding the first observation)
        images = _sample["observation"]["image_primary"]
        images = einops.rearrange(
            images, "B T H W C -> B (T C) H W"
        )  # remove cond_steps dimension
        texts = [
            text.decode("utf-8") for text in _sample["task"]["language_instruction"]
        ]
        actions = _sample["action"]
        proprios = _sample["observation"]["proprio"]

        # check padding
        if not _sample["observation"]["timestep_pad_mask"].all():
            print("Padding for history obs past trajectory start")
        if not _sample["action_pad_mask"].all():
            print("Padding for action chunks past trajectory end")

        # verify the normalization
        if actions.abs().max() > 1 or proprios.abs().max() > 1:
            breakpoint()
        cnt_batch += 1
    load_time = time.time()
    print(f"Iterative over {cnt_batch} batches: {load_time - prep_time:.2f}s")
