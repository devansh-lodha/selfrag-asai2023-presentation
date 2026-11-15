# DeepSpeed ZeRO-3 and Finetune Execution

## 1. Plain-English Intuition: Why Training a 7B or 13B Model Requires Sharding Across GPUs

At inference time, loading a Llama-2 7B model ($P = 7 \times 10^9$ parameters) in 16-bit precision (`2 bytes` per parameter) requires only $7 \times 10^9 \times 2\text{ bytes} = 14\text{ GB}$ of GPU memory, which fits comfortably on a single $24\text{ GB}$ GPU.

Why, then, does full-parameter training of Llama-2 7B or 13B require multiple $80\text{ GB}$ A100 GPUs and DeepSpeed ZeRO Stage 3?

Let us count the exact number of bytes required for every trainable parameter $\theta_i$ during mixed-precision (`bfloat16`) training with the AdamW optimizer:
1. 16-bit model weight ($\theta_i^{\text{bf16}}$, $2\text{ bytes}$): used in forward and backward matrix multiplications.
2. 16-bit gradient ($g_i^{\text{bf16}}$, $2\text{ bytes}$): computed during backpropagation.
3. 32-bit master weight ($\theta_i^{\text{fp32}}$, $4\text{ bytes}$): maintained by the optimizer because adding tiny learning-rate updates ($\eta = 2 \times 10^{-5}$) directly to a 16-bit number would round small updates down to zero.
4. 32-bit Adam first moment ($m_i^{\text{fp32}}$, $4\text{ bytes}$) and 32-bit Adam second moment ($v_i^{\text{fp32}}$, $4\text{ bytes}$): $8\text{ bytes}$ total to track the running mean of gradients $g_i$ and squared gradients $g_i^2$.

Adding these up gives $2 + 2 + 4 + 4 + 4 = 16\text{ bytes}$ of static memory per parameter:

$$\text{Static Memory (Standard DDP)} = 16 \times P\text{ bytes}$$

- For Llama-2 7B ($P = 7 \times 10^9$): $16 \times 7\text{ GB} = 112\text{ GB}$ (already exceeding an $80\text{ GB}$ A100 before storing any forward-pass activations!).
- For Llama-2 13B ($P = 13 \times 10^9$): $16 \times 13\text{ GB} = 208\text{ GB}$!

In standard Distributed Data Parallel (DDP), every GPU holds a complete duplicate copy of all $112\text{ GB}$ (or $208\text{ GB}$). DeepSpeed's Zero Redundancy Optimizer (ZeRO) eliminates this duplication across $N_{\text{GPU}}$ GPUs in three stages:

| Memory Component | Bytes per Parameter | Standard DDP (Per GPU) | ZeRO Stage 1 (Shards Optimizer) | ZeRO Stage 2 (+ Shards Gradients) | ZeRO Stage 3 (+ Shards Weights) |
|---|---|---|---|---|---|
| FP32 Master Weights + Adam $m, v$ | $12\text{ bytes}$ | $12 P$ | $\dfrac{12 P}{N_{\text{GPU}}}$ | $\dfrac{12 P}{N_{\text{GPU}}}$ | $\dfrac{12 P}{N_{\text{GPU}}}$ |
| BF16 Gradients | $2\text{ bytes}$ | $2 P$ | $2 P$ | $\dfrac{2 P}{N_{\text{GPU}}}$ | $\dfrac{2 P}{N_{\text{GPU}}}$ |
| BF16 Model Weights | $2\text{ bytes}$ | $2 P$ | $2 P$ | $2 P$ | $\dfrac{2 P}{N_{\text{GPU}}}$ |
| Total for 7B on $N_{\text{GPU}} = 4$ GPUs | $16\text{ bytes}$ | $112\text{ GB}$ | $49\text{ GB}$ | $38.5\text{ GB}$ | $28\text{ GB}$ |
| Total for 13B on $N_{\text{GPU}} = 4$ GPUs | $16\text{ bytes}$ | $208\text{ GB}$ | $91\text{ GB}$ | $71.5\text{ GB}$ | $52\text{ GB}$ |

Under ZeRO Stage 3 on $N_{\text{GPU}} = 4$ GPUs, each GPU permanently stores only $1/4$ of the optimizer states, $1/4$ of the gradients, and $1/4$ of the 16-bit weights ($28\text{ GB}$ for 7B, $52\text{ GB}$ for 13B). When layer $\ell$ is about to execute its forward or backward pass, all $4$ GPUs broadcast (`all-gather`) their $1/4$ shards of layer $\ell$'s weights, compute layer $\ell$, and immediately discard the non-local weight shards before moving to layer $\ell + 1$.

---

## 2. Walkthrough of `stage3_no_offloading_accelerate.conf`

Let us inspect `reference/code/retrieval_lm/stage3_no_offloading_accelerate.conf`:

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 1e5,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

Every setting in this file controls a specific part of the ZeRO-3 runtime:
- `"bf16": {"enabled": "auto"}`: Uses `bfloat16` mixed precision (set via `--mixed_precision bf16` in the launch script). Unlike `float16` (which has only $5$ exponent bits and overflows at $65504$), `bfloat16` has $8$ exponent bits (the exact same dynamic range as `float32`), preventing gradient underflow and overflow.
- `"stage": 3`: Partitions optimizer states, gradients, and model parameters across GPUs without CPU offloading (`no_offloading`, keeping all shards in fast GPU HBM).
- `"overlap_comm": true`: Overlaps the inter-GPU `all-gather` and `reduce-scatter` communication of layer $\ell + 1$ with the GPU matrix multiplication of layer $\ell$.
- `"contiguous_gradients": true`: Writes gradients into a pre-allocated contiguous buffer during backpropagation to avoid memory fragmentation.
- `"stage3_max_live_parameters": 1e9`: Caps the maximum number of unsharded parameters held in GPU memory at any instant at $10^9$ parameters ($2\text{ GB}$).
- `"stage3_gather_16bit_weights_on_model_save": true`: Because each GPU only owns $1/4$ of the weights during training, this flag tells DeepSpeed to gather all shards across the GPUs into a single complete model checkpoint when saving.

---

## 3. Walkthrough of `script_finetune_7b.sh` and `script_finetune_13b.sh`

Let us examine `reference/code/retrieval_lm/script_finetune_7b.sh` (`script_finetune_13b.sh` is identical except for `MODEL_SIZE=13B`, `meta-llama/Llama-2-13b-hf`, and `--max_seq_length 1536`). Both scripts use $4$ GPUs, matching Appendix B.1 ("4 Nvidia A100 with 80GB memory"), whereas the main README says the authors used $8$ A100s with $40\text{ GB}$ for 7B and $4$ A100s with $80\text{ GB}$ for 13B:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_flash_attn \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --train_file train.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir output/self_rag_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens
```

### 3.1 Effective Batch Size and Gradient Accumulation Math

Fitting a $2048$-token sequence through a 7B or 13B transformer requires setting the instantaneous per-GPU batch size to $B_{\text{micro}} = 1$ (`BATCH_SIZE_PER_GPU=1`). However, updating the weights with a noisy batch size of $1 \times 4 = 4$ examples would make training unstable.

To achieve a target effective batch size of $B_{\text{total}} = 128$ (`TOTAL_BATCH_SIZE=128`) across $N_{\text{GPU}} = 4$ GPUs, the script computes the number of gradient accumulation steps $G_{\text{acc}}$:

$$G_{\text{acc}} = \frac{B_{\text{total}}}{N_{\text{GPU}} \times B_{\text{micro}}} = \frac{128}{4 \times 1} = 32$$

During training, each GPU runs $32$ sequential forward and backward passes of $1$ example each, adding up the gradients in memory:

$$g_{\text{accum}} = \frac{1}{N_{\text{GPU}} \cdot G_{\text{acc}}} \sum_{w=1}^{N_{\text{GPU}}} \sum_{s=1}^{G_{\text{acc}}} \nabla_{\mathcal{M}} \mathcal{L}_{w, s}$$

Only after all $32$ micro-steps ($4 \times 32 = 128$ examples total) have accumulated does the optimizer take $1$ weight update step.

### 3.2 Sequence Length Difference Between 7B and 13B

Notice the only hyperparameter difference between `script_finetune_7b.sh` and `script_finetune_13b.sh`:
- For 7B (`script_finetune_7b.sh` line 22): `--max_seq_length 2048` (leaving $\sim 52\text{ GB}$ of an $80\text{ GB}$ A100 for activation memory after the $28\text{ GB}$ ZeRO-3 static state).
- For 13B (`script_finetune_13b.sh` line 22): `--max_seq_length 1536` (reported as $1,524$ in Appendix B.1 of the paper, reduced because 13B's $52\text{ GB}$ ZeRO-3 static state leaves only $\sim 28\text{ GB}$ for activations on an $80\text{ GB}$ A100).

---

## 4. Line-by-Line Walkthrough of the Training Loop in `finetune.py` (Lines 455-635)

Let us trace the four key sections of the training loop in `reference/code/retrieval_lm/finetune.py`.

### 4.1 Optimizer Weight Decay Grouping (Lines 455-469)

```python
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
```

In transformer training, 1D bias vectors and LayerNorm scaling weights should not be shrunk toward zero by $\ell_2$ weight decay (`"weight_decay": 0.0`), while 2D weight matrices receive `args.weight_decay` (note that in `script_finetune_7b.sh`, `--weight_decay 0.` sets weight decay to $0.0$ for both groups).

For Llama-2, however, the second group is empty: its linear layers have no biases, and its RMSNorm weights are named `input_layernorm.weight`, `post_attention_layernorm.weight`, and `norm.weight`, none of which contains the substring `"layer_norm.weight"`. With nonzero weight decay, the norm weights would be decayed too; with `--weight_decay 0.` the mismatch has no effect.

### 4.2 Multi-Process Learning Rate Scheduler Scaling (Lines 471-505)

Look at how `finetune.py` calculates the number of warmup and total steps for `get_scheduler` before and after `accelerator.prepare`:

```python
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
```

Why is `args.max_train_steps` calculated twice (once before `accelerator.prepare` on line 475, and again after `accelerator.prepare` on line 501)?
1. Before `accelerator.prepare` (line 473): `train_dataloader` has not been split across the $N_{\text{GPU}} = 4$ GPUs yet, so `len(train_dataloader)` is $4\times$ larger than the per-GPU batch count.
2. When `accelerator.prepare` wraps `lr_scheduler` on line 495, Hugging Face `Accelerate` steps the underlying PyTorch scheduler $N_{\text{GPU}} = 4$ times on every optimization step! Therefore, `get_scheduler` (line 486) needs `num_training_steps_for_scheduler` to be $4\times$ the true optimization step count (which `args.max_train_steps` on line 475 already is when `overrode_max_train_steps` is `True`).
3. After `accelerator.prepare` shards `train_dataloader` across the $4$ GPUs, `len(train_dataloader)` is divided by $4$, so line 501 recalculates `args.max_train_steps` to equal the true number of optimization steps. If the training file holds the $145,619$ examples reported in Table 3, that is about $\frac{145619}{128} \times 3 \approx 3413$ total steps, with $3\%$ warmup $\approx 102$ steps (Module 07 Part 3 explains why the postprocessed $\mathcal{D}_{\text{gen}}$ can be smaller than its input).

### 4.3 Gradient Accumulation Loop and Saving ZeRO-3 Checkpoints (Lines 551-635)

Inside the epoch loop (lines 551-605):

```python
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
```

- `use_cache=False` disables storing key-value caches (which are only useful during token-by-token test-time generation and would waste GPU memory during teacher-forcing training) and is required by the FlashAttention monkey patch (`assert not use_cache` in `llama_flash_attn_monkey_patch.py` line 55).
- `with accelerator.accumulate(model):` intercepts `optimizer.step()`, `optimizer.zero_grad()`, and `lr_scheduler.step()` so they only execute once every `gradient_accumulation_steps` ($32$) micro-batches, when `accelerator.sync_gradients` becomes `True`.
- Finally, when training finishes (lines 613-635), `state_dict = accelerator.get_state_dict(model)` gathers the ZeRO-3 sharded weights across all GPUs, and `unwrapped_model.save_pretrained` writes the complete trained Self-RAG Generator $\mathcal{M}$ and expanded tokenizer to `args.output_dir`.
