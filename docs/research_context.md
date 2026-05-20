# Research Context

The project implements the computational track for a retina-inspired edge
detection paper.

Core biological observation to test:

Retinal ganglion cell population responses to a moving edge are modulated by a
nearby stationary luminance boundary.

Core model principle:

Local edge response should be modulated by nearby boundary context.

RBCM translates this principle into three parts:

1. A local edge branch that extracts local edge features.
2. A boundary context branch that extracts nearby boundary and region context.
3. A signed modulation gate in `[-1, 1]` that can enhance, suppress, or keep the
   local edge feature nearly unchanged.

Main model formula:

```text
F_out = F_local + alpha * F_local * G_context
```

The first engineering goal is to keep the code simple enough to adjust while MEA
results are still being explored.
