# pg-vla

## Installation
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the dependencies will be configured automatically when running any `uv run ...` command. Or `pip install -e .`

### Tests
VLA with real img/text, and output text
```console
uv run src/model/vla/model.py --text_only --load_pretrained_weights
```

VLA with real img/text, and output text while also performing cross attention between blocks (should generate the same text as above but slower)
```console
uv run src/model/vla/model.py --text_only --text_from_joint_model --load_pretrained_weights
```

VLA with dummy img/text, proprio, and action, output flow matching action
```console
uv run src/model/vla/model.py
```

Block attention with dummy embeddings for img/text, proprio, and action
```console
uv run src/model/vla/modules.py
```

Text generation with original implementaton
```console
git clone https://huggingface.co/google/paligemma2-3b-pt-224    # at desired cache location
uv run scripts/tests/run_paligemma.py \
    --model_path "/n/fs/llm-unc/.cache/paligemma-3b-pt-224" \
    --prompt "this image shows " \
    --image_file_path "media/maniskill_pp.png" \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9
```

## Data

### OXE
Currently using meta robot and widowx data (fractal/RT-1, bridge, optionally BC-Z and RoboVQA)

Download data by
```console
uv run gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0 ../data   # bridge_dataset
```

Preprocess (taken from [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main) and [Octo](https://github.com/kpertsch/rlds_dataset_mod/blob/main/prepare_open_x.sh)). First we need to comment out Line 299-306 in `.venv/lib/python3.10/site-packages/tensorflow_datasets/core/dataset_builder.py` to avoid the `AttributeError: 'MultiplexedPath' object has no attribute 'parts'` error (seems an issue with running python3.10; using `tensorflow_datasets==4.9.2` fixes this issue but disabling gcs does not work any more somehow). Then run (with enough RAM)
```console
uv run scripts/data/modify_rlds_dataset.py \
    --dataset=fractal20220817_data \
    --data_dir=/n/fs/llm-unc/data/ \
    --target_dir=/n/fs/llm-unc/data/resize_224 \
    --mods=resize_and_jpeg_encode \
    --n_workers=40 \
    --max_episodes_in_memory=400
```
This resizes the images to 224x224 (as opposed to 256x256 in Octo).

Running dataloader for the first time will take some time for calculating the normalization statistics, which is then saved.
```console
uv run src/data/dataloader.py \
    --data_path=/n/fs/llm-unc/data/resize_224 \
    --mix=oxe_simple
```

## Training

See launch file
```console
uv run torchrun --nnodes=1 --nproc_per_node=$NUM_GPU --rdzv_id=100 --rdzv_backend=c10d --max-restarts=1 --standalone --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT scripts/run.py
```

If using quantization, need to modify Line 474 in `.venv/lib64/python3.10/site-packages/bitsandbytes/autograd/_functions.py` to `return output.clone` from `return output` ([related issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/736)).


## Resources

Octo, OpenVLA, Pi0 paper

Paligemma [https://github.com/hkproj/pytorch-paligemma/tree/main](https://github.com/hkproj/pytorch-paligemma/tree/main)
