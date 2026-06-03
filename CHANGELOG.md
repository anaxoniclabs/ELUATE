# Changelog

All notable changes to ELUATE are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and ELUATE adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
on the public surface defined in [`docs/api.md`](docs/api.md).

## [0.0.1] - 2026-06-03

First release. Targets CUDA / Google Colab as the primary platform;
macOS Apple Silicon is supported as the local development and testing
target.

### Added

- **CLI** - `eluate video.mp4` strips background music from a video and
  writes a new MP4 next to it. Single-file, batch (`--batch files.txt`),
  and folder (`--folder dir/`) modes; `eluate info` reports device,
  FFmpeg, and model status.
- **Python API** - `eluate.elute()`, `eluate.Session`, `eluate.Result`,
  and a typed exception hierarchy under `eluate.EluateError`. Stable
  contract documented in [`docs/api.md`](docs/api.md); semver applies to
  this surface only.
- **Streaming demix path** - fixed-size ring buffer in
  `eluate.core.separator._demix_streaming` keeps peak memory bounded on
  long inputs. ~2.15x less peak memory than the upstream Bandit v2
  reference on an 84-minute documentary; see
  [`docs/bench/`](docs/bench/) for raw numbers and methodology.
- **Bandit v2 model** with per-language checkpoints (`multi`, `eng`,
  `deu`, `fra`, `spa`, `cmn`, `fao`) downloaded from Zenodo record
  `12701995`. The `multi` checkpoint is SHA-256 verified at install
  time; other variants currently download without integrity checks (a
  warning is printed).
- **Bundled inference subset** of ZFTurbo's
  [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
  framework under `eluate/_vendor/` so a `pip install eluate` works
  without the upstream submodule.
- **Colab template** at `notebooks/eluate_colab_template.ipynb` for
  running the Python API on a GPU runtime.
- **Linux memory reporting** in `eluate info` via `/proc/meminfo`, and
  an OS-aware FFmpeg install hint (`apt-get` on Linux, `brew` on macOS).
- **Local debug log** - opt-in JSONL telemetry written to
  `~/.eluate/telemetry.jsonl`. Off by default; enable with
  `ELUATE_TELEMETRY=1`.
- **Trusted-publisher PyPI release pipeline** with PEP 740 attestations
  and SLSA build provenance.

### Known limitations

- Bandit v2 weights are CC-BY-SA 4.0; commercial use carries
  share-alike obligations. See the README "Licensing" section.
- Non-`multi` Bandit v2 checkpoints currently lack verified SHA-256
  digests.

[0.0.1]: https://github.com/anaxoniclabs/ELUATE/releases/tag/v0.0.1
