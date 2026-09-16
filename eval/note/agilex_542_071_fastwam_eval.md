# FastWAM AgileX 50-window Eval

Code:

```bash
cd /home/chw/code/packages/FastWAM
CUDA_VISIBLE_DEVICES=0 bash eval/code/run_agilex_50w_eval.sh
```

The launch script activates conda env `fastwam`.

Defaults:

- requested dataset: `/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_071`
- resolved fallback dataset if requested path is missing: `/mnt/data/dataset/ei/huggingface/modanqing/agilex_empty_the_box_all_542_0711`
- checkpoint: `/mnt/data/chw/fastwam/runs/agilex_empty_box_giga_init_bs8_8gpu_100k/checkpoints/weights/step_010000.pt`
- output: `/mnt/data/chw/fastwam/evaluate_results/agilex_empty_box_giga_init/eval_step_010000_50w`

Use `--output-dir /some/path` to override the output location.

Useful dry checks:

```bash
python eval/code/eval_agilex_fastwam.py --stage check
python eval/code/eval_agilex_fastwam.py --stage data
python eval/code/eval_agilex_fastwam.py --stage smoke --max-windows 1
```

The script is intentionally single-process and single-visible-GPU by default.
It runs action-only inference with `batch_size=1`, `num_workers=0`, text embedding
cache enabled, and 4 denoising steps to reduce interference with existing GPU jobs.
