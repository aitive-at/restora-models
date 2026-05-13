# coliraz v1 (archived)

This is the v1 colorization-only project. It is preserved unchanged.
To resurrect:

```sh
cd legacy/coliraz-v1
uv sync --extra dev
uv run coliraz train --config configs/laion-large-vivid.yaml --data <path>
```

For the active multi-task v2, see ../../README.md.
