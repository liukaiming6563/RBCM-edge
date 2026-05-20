# RBCM-edge

RBCM-edge is the initial engineering scaffold for the Neurocomputing paper idea
"Retina-inspired boundary-context modulation for generalized edge detection".

The project keeps two related but separate tracks:

1. MEA analysis for retinal ganglion cell responses under `bar`, `single_edge`,
   and `double_edge` stimulus paradigms.
2. Edge detection model code centered on the Retinal Boundary-Context Modulation
   (RBCM) module.

## Local Data Layout

Large data are intentionally not tracked by git. The expected local MEA data root is:

```text
D:\study\project\RBCM-Edge\MEA_data
```

Expected contents:

```text
MEA_data/
  000031/ ... 000039/      # Kilosort outputs for nine experiments
  sti_info/
    bar/
    single_edge/
    double_edge/
```

The default experiment mapping is:

```text
000031 double_edge
000032 single_edge
000033 bar
000034 double_edge
000035 single_edge
000036 bar
000037 double_edge
000038 single_edge
000039 bar
```

## Argparse Pattern

All runnable scripts use a `DEFAULT_ARGS` dictionary near the top of the file.
You can either pass arguments from the command line or edit the default values
inside the script and run it directly from an IDE.

Example:

```powershell
$env:PYTHONPATH="D:\study\project\RBCM-Edge\src"
python scripts\mea\inspect_mea_tree.py --data-root MEA_data
```

## First Commands

Inspect the local MEA tree:

```powershell
$env:PYTHONPATH="D:\study\project\RBCM-Edge\src"
python scripts\mea\inspect_mea_tree.py
```

Build first response tables:

```powershell
$env:PYTHONPATH="D:\study\project\RBCM-Edge\src"
python scripts\mea\build_response_tables.py --run-id all
```

Smoke-check the RBCM model module:

```powershell
$env:PYTHONPATH="D:\study\project\RBCM-Edge\src"
python scripts\model\run_rbcm_smoke_check.py
```

Run normal script-style checks in PyCharm or PowerShell:

```powershell
$env:PYTHONPATH="D:\study\project\RBCM-Edge\src"
python scripts\checks\check_mea_metrics.py
python scripts\checks\check_rbcm_shapes.py
```

This project intentionally keeps checks under `scripts/checks/` as normal Python files, so
PyCharm should offer normal Python run configurations for checks and scripts.

## Edge Model Pipeline

The edge detection training pipeline lives under `edge_model/`. It contains
dataset loading, transforms, training, evaluation, inference, metrics, prediction
visualization, and gate heatmap saving.

For local 3070Ti debugging, start with:

```powershell
$env:PYTHONPATH="D:\study\project\RBCM-Edge\src"
python edge_model\tools\inspect_edge_data.py
python edge_model\tools\check_dataloader.py
python edge_model\train.py --config edge_model\configs\local_3070ti.yaml --epochs 2 --batch-size 2
python edge_model\tools\analyze_training_log.py
```

Generated checkpoints, logs, metrics, predictions, gate heatmaps, and visual
comparisons are saved under:

```text
outputs/edge_detection/<experiment_name>/
```

When `train.auto_plot_log: true` is enabled, training automatically saves loss
and ODS/OIS/AP trend plots under:

```text
outputs/edge_detection/<experiment_name>/plots/
```
