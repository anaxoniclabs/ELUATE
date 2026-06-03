# ELUATE

As a muslim, I've experienced this friction many times: the
documentary I want to watch or some informative long-form video I want to learn from,
almost always comes with a score running underneath. Normally my next step would be to just not watch it. However, on one of those days, I just thought maybe I could build a simple tool that would take a video in and hand me back the same video with the music gone.

ELUATE is that simple thing. It is a CLI that you give it a video file, it removes the music, keeps the dialogue and sound effects, and outputs the video. The video stream is copied through untouched so it is bit-for-bit unchanged.

In terminal you do

```bash
eluate documentary.mp4
# → ~/Documents/ELUATE/documentary_eluted.mp4
```

I also wanted this to have a small API with the same engine behind, so I can
wire ELUATE into other tools (a content pipeline, a batch job,
my other project, etc.)

```python
import eluate

eluate.elute("documentary.mp4")
```


I wanted to share this here for muslims who run into the same friction trying to
learn, research, experiment. However as I was developing this and doing research about it, I found that the benefit can be much broader. For instance people with hearing loss where the score fights the narration, or someone with focus or auditory-processing conditions where the score becomes another competing thing you have to filter out. 

If that's you too, I hope this tool helps you.

It's a one-maintainer project + AI. Feel free to fork and develop, I am sure there are much more talented and knowledgeable muslims than me that could make this much much better.

## Install

```bash
pip install eluate
```

ELUATE needs FFmpeg on your PATH. It's preinstalled on Colab; on Linux
install it with `sudo apt-get install -y ffmpeg`, on macOS with
`brew install ffmpeg`.

The first run downloads a ~450 MB model checkpoint from
[Zenodo](https://zenodo.org/records/12701995) into `~/.eluate/`.
It only happens once.

```bash
eluate info   # verify device, FFmpeg, and model are ready
```

A ready-to-run Colab notebook is at
[`notebooks/eluate_colab_template.ipynb`](notebooks/eluate_colab_template.ipynb):
set the runtime to a GPU (`Runtime -> Change runtime type -> GPU`), then
run the cells top to bottom.

## Usage

```bash
eluate                                # interactive mode (prompts for file)
eluate video.mp4                      # single file → ~/Documents/ELUATE/
eluate video.mp4 -o custom.mp4        # custom output path
eluate --checkpoint eng video.mp4     # use the English-optimised model
eluate setup                          # download the default model for API use
eluate --batch files.txt              # process paths from a list file
eluate --folder /path/to/videos       # process every video in a folder
eluate --folder ./videos --batch-size 10   # folder in batches of 10
eluate info                           # system / model status
eluate video.mp4 --device cpu         # force CPU, skip MPS
eluate video.mp4 --force              # skip duration / disk-space checks
```

Supported input containers: `mp4`, `mkv`, `avi`, `mov`, `webm`, `flv`,
`wmv`, `m4v`.

### Model

ELUATE ships a single model: **Bandit v2**, CC-BY-SA 4.0, 48 kHz.
CC-BY-SA permits commercial use under share-alike terms; see
[Licensing](#licensing) for the caveats before shipping anything
commercial.

Bandit v2 ships per-language checkpoints (`multi`, `eng`, `deu`, `fra`,
`spa`, `cmn`, `fao`). The default is `multi`. Swap with
`--checkpoint eng` etc.

### Python API

For batch loops or wiring ELUATE into your own code, instantiate a
`Session` so the model loads once and is reused:

```python
import eluate

with eluate.Session() as session:
    for path in ["a.mp4", "b.mp4", "c.mp4"]:
        session.elute(path)
```

`eluate.elute()`, `eluate.Session`, `eluate.Result`, and the typed
exception hierarchy under `eluate.EluateError` form the entire v1.x
public surface; semver applies to those names only. Full reference,
including progress callbacks and stem-only outputs, in
[`docs/api.md`](https://github.com/anaxoniclabs/ELUATE/blob/main/docs/api.md).

## How it works

With ([Bandit v2](https://arxiv.org/abs/2407.07275)) ELUATE splits the audio into speech, music, and sfx, discards the music stem, mixes the other two back together, and then into a copy of the video.

```
 input.mp4 ─┬─▶ ffmpeg audio extract ─▶ 48 kHz WAV
            │
            │                            Bandit v2 (streaming demix)
            │                                  │
            │                    ┌─────────────┼─────────────┐
            │                    ▼             ▼             ▼
            │                  speech        music          sfx
            │                    │           drop            │
            │                    └────────── mix ────────────┘
            │                                │
            └─▶ ffmpeg mux  ◀─────  new audio track (speech + sfx)
                   │
                   ▼
           output_eluted.mp4   (video stream copied as-is)
```

The separator uses a **streaming demix** path: a fixed-size ring buffer
and virtual padded-chunk construction so the full audio track is never
held in memory. See [Testing](#testing) for the memory benchmark.

## Testing

Even though over time I started to optimize the code towards CUDA for people to maybe run this on Colab, I've developed it on a Mac mini, so testing was largely done with that.

On an 84-minute documentary on my M4 with 24 GB, it had 19 GB of peak memory usage with ELUATE latest state compared to 41 GB for the original (non-optimized) upstream reference, which finished with macOS swapping about 20 GB to disk, meaning ELUATE used 2.15× less in practice. To do that ELUATE uses a fixed-size ring buffer instead, processing the audio in a moving window without ever holding the full track in memory.
Check [`docs/bench/`](https://github.com/anaxoniclabs/ELUATE/blob/main/docs/bench/) for the plain-language
summary, full methodology in
[`memory-benchmark.md`](https://github.com/anaxoniclabs/ELUATE/blob/main/docs/bench/memory-benchmark.md), and a
windowing bug this benchmarking uncovered and fixed.


It isn't faster than [Demucs](https://github.com/facebookresearch/demucs).
I optimised the memory path for long files, not raw throughput, and
you'll feel that on short inputs where Demucs finishes first. However in terms of use case, as far as I know, Demucs is more of a music stem separation model.

The CLI doesn't give you the stems. ELUATE's terminal workflow is
video in, video out; the Python API can export speech and sfx WAVs for
downstream tools, but it still never exposes the music stem.

## Configuration and data

ELUATE writes data to two places, both under your home directory:

| Path | What lives there |
|---|---|
| `~/.eluate/models/` | Downloaded model checkpoints (~450 MB each), configs |
| `~/.eluate/venv/` | Python virtual env (created by `install.sh`) |
| `~/.eluate/telemetry.jsonl` | Local debug log (only if you enable it) |
| `~/Documents/ELUATE/` | Processed output videos |

Nothing is sent over the network after the initial model download.

### Local debug log (off by default)

ELUATE can write a local JSONL log of processing stages to help debug performance issues. **The
log never leaves your machine.** It's a plain file you can read,
delete, or ignore.

Telemetry is **off by default.** Enable it when you want a paper trail:

```bash
ELUATE_TELEMETRY=1 eluate video.mp4
```

The log contains nothing that identifies you or your files. Delete it any
time.

## Licensing

- **ELUATE's own code**: MIT (see [`LICENSE`](LICENSE)).
- **Bandit v2 model weights**: CC-BY-SA 4.0, from
  [Zenodo 12701995](https://zenodo.org/records/12701995). CC-BY-SA
  permits commercial use under **share-alike** terms: any derivative
  work you distribute under these weights must itself be licensed
  CC-BY-SA. That clause is operationally hostile to a lot of commercial
  software (it arguably extends to downstream derivative works).
  **Consult a lawyer before shipping a commercial product built on
  ELUATE.** The README can't and doesn't give legal advice.
- **Vendored separator framework** at `vendor/mss-training/`: ZFTurbo's
  [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
  MIT-licensed.


## Contributing

Open an issue or PR at
<https://github.com/anaxoniclabs/ELUATE/issues>.

For bug reports, running with the debug log enabled would help a lot:

```bash
ELUATE_TELEMETRY=1 eluate your-video.mp4
# then attach ~/.eluate/telemetry.jsonl (it never leaves your machine until you share it)
```

## Acknowledgements

- [Karn N. Watcharasupat, Chih-Wei Wu, and Iroro Orife](https://arxiv.org/abs/2407.07275)
  for the Bandit v2 architecture and the Divide-and-Remaster v3 dataset.
- [ZFTurbo](https://github.com/ZFTurbo) for the
  Music-Source-Separation-Training framework that ELUATE's separator is
  built on.
