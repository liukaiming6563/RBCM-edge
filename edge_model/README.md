# Edge Model Engineering

This folder contains the runnable edge detection engineering pipeline for
RBCM-edge. It is intentionally independent from the MEA analysis scripts, but it
reuses the RBCM modules under `src/rbcm_edge`.

## Local 3070Ti Demo

Run these commands from the project root:

```powershell
$env:PYTHONPATH="D:\study\project\RBCM-Edge\src"
python edge_model\tools\inspect_edge_data.py
python edge_model\tools\check_dataloader.py
python edge_model\train.py --config edge_model\configs\local_3070ti.yaml --epochs 2 --batch-size 2
```

## Output Layout

Training and evaluation outputs are created under:

```text
outputs/edge_detection/<experiment_name>/
  checkpoints/
  logs/
  metrics/
  predictions/
  gate_heatmaps/
  visualizations/
```

The `outputs/` folder is ignored by git. Keep generated predictions,
checkpoints, and visualizations local or copy them manually when preparing paper
figures.

## Script Style

Each runnable file has a `DEFAULT_ARGS` dictionary near the top. In PyCharm you
can edit those values and run the file directly, or you can override them from
the command line.
