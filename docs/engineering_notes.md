# Engineering Notes

## Coding Style

All analysis scripts should keep two usage modes:

1. External control through argparse.
2. Direct editing of the `DEFAULT_ARGS` dictionary at the top of the script.

This is deliberate: MEA analysis changes frequently during exploration, while
final experiments need reproducible command lines.

## Data Policy

Do not commit `MEA_data/`, model checkpoints, predictions, or generated figures.
Commit code, configs, and lightweight documentation only.

## MEA Analysis Priority

1. Inspect Kilosort and stimulus files.
2. Build per-event spike count and firing rate tables.
3. Add baseline-corrected response metrics.
4. Add single-edge vs boundary-context matching and modulation index.
5. Add distance-to-boundary and population decoding once event geometry is stable.
