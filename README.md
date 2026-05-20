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

Smoke-test the RBCM model module:

```powershell
$env:PYTHONPATH="D:\study\project\RBCM-Edge\src"
python scripts\model\smoke_test_rbcm.py
```
