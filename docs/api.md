# ELUATE Python API

`eluate` ships a small, stable Python API alongside its CLI. Import the
package and call one function for a one-shot run, or instantiate a
`Session` to amortise model loading across many files.

```python
import eluate

result = eluate.elute("documentary.mp4")
print(result.video)  # PosixPath('.../documentary_eluted.mp4')
```

## Quick reference

| Name                                 | Kind        | Purpose                                              |
|--------------------------------------|-------------|------------------------------------------------------|
| [`elute`](#eluate-elute)               | function    | One-shot: load model, process one file, tear down.   |
| [`Session`](#class-eluate-session)   | class       | Reusable session that loads the model once.          |
| [`Result`](#class-eluate-result)     | dataclass   | Frozen return value of every successful run.         |
| [`EluateError`](#exceptions)         | exception   | Base class for eluate-specific failures.             |
| `DurationOutOfRange`                 | exception   | Input duration is zero, negative, or above the cap.  |
| `InsufficientDiskSpace`              | exception   | Target filesystem lacks free space for the run.      |
| `ModelNotInstalledError`             | exception   | Required checkpoint is not on this machine.          |

Everything else under `eluate.*` (`eluate.core`, `eluate.utils`,
`eluate.ui`) is internal and may change in any release.

---

## Why an API (and not just the CLI)

The CLI is correct for humans; it is the wrong shape for code. Calling
`eluate` via `subprocess` means arguments pass as strings, errors come
back as exit codes, the model reloads from disk on every invocation,
and any downstream tool that wants the audio (Whisper, loudness
analysis, classifiers) has to re-extract it with `ffmpeg`.

The Python API addresses all four. Inputs are `pathlib.Path` objects;
the return value is a frozen dataclass (`Result`); failures raise
typed exceptions; diagnostics flow through
`logging.getLogger("eluate")`. A `Session` loads the model once and
reuses it across many `.elute()` calls, so a batch loop runs in minutes
instead of hours. The `outputs` keyword can request `"speech"` and/or
`"sfx"` directly, returning WAV paths in the `Result` without an
extra `ffmpeg` pass. The public surface is locked: only the four
names at the top of this document, plus their public attributes, are
covered by semver; everything else under `eluate.*` is internal.

---

## Installation

Install eluate and its dependencies (see the project README for full
setup, including the FFmpeg requirement and the first-run model
download). Once installed:

```python
import eluate
eluate.__version__  # '0.0.1'
```

---

## Quickstart

### One file

```pycon
>>> import eluate
>>> result = eluate.elute("documentary.mp4")
>>> result.video
PosixPath('/cwd/documentary_eluted.mp4')
>>> result.processing_time
312.4
```

By default this writes `documentary_eluted.mp4` to the current working
directory and returns a [`Result`](#class-eluate-result).

### Many files (recommended)

```python
import eluate

with eluate.Session() as session:
    for path in ["a.mp4", "b.mp4", "c.mp4"]:
        session.elute(path)
```

The model loads on the first `.elute()` call and is reused for the rest
of the session.

### Stems for downstream tools

```python
import eluate

result = eluate.elute("interview.mp4", outputs=("speech",))
# result.speech → PosixPath('.../interview_speech.wav')
# result.video  → None  (not requested)
```

---

## API reference

<a id="eluate-elute"></a>
### `eluate.elute(input, *, outputs=("video",), output_dir=None, overwrite=False, force=False, audio_codec="aac", audio_bitrate="256k", on_progress=None, device=None, checkpoint="multi") -> Result`

One-shot equivalent of `Session(...).elute(input, outputs=outputs)`.
Loads the model, processes one file, and tears down. Use this for a
single file; use [`Session`](#class-eluate-session) for two or more.

**Parameters**

| Name            | Type                              | Default       | Description |
|-----------------|-----------------------------------|---------------|-------------|
| `input`         | `str \| pathlib.Path`             | required      | Path to the input video. |
| `outputs`       | `Iterable[str]`                   | `("video",)`  | Any combination of `"video"`, `"speech"`, `"sfx"`. `"music"` is rejected; eluate is a removal tool, not a stem separator. |
| `output_dir`    | `str \| Path \| None`             | `None`        | Where outputs land. `None` means the current working directory. Created if missing. |
| `overwrite`     | `bool`                            | `False`       | If `False`, raises `FileExistsError` when any requested output already exists. |
| `force`         | `bool`                            | `False`       | Bypass the duration validator and disk-space precheck (matches CLI `--force`). Does **not** affect overwrite policy. |
| `audio_codec`   | `str`                             | `"aac"`       | FFmpeg audio codec for the muxed video output. |
| `audio_bitrate` | `str`                             | `"256k"`      | FFmpeg audio bitrate for the muxed video output. |
| `on_progress`   | `Callable[[float, str], None]`    | `None`        | Progress callback. Called with `(fraction, stage)` where `fraction` is monotonic in `[0.0, 1.0]` and `stage` is one of `"extract"`, `"load_model"`, `"separate"`, `"compile"`. |
| `device`        | `"cuda" \| "mps" \| "cpu" \| None`| `None`        | Inference device. `None` auto-detects (CUDA > MPS > CPU). |
| `checkpoint`    | `str`                             | `"multi"`     | Bandit-v2 language-variant checkpoint. Valid values: `eluate.utils.paths.CHECKPOINT_KEYS`. |

**Returns**

- [`Result`](#class-eluate-result): frozen dataclass with the produced
  paths, input duration, and wall-clock processing time.

**Raises**

- `FileNotFoundError`: `input` does not exist.
- `FileExistsError`: a requested output exists and `overwrite=False`.
- `DurationOutOfRange`: input is zero-length or longer than the cap;
  bypass with `force=True`.
- `InsufficientDiskSpace`: target filesystem cannot fit the run;
  bypass with `force=True`.
- `ModelNotInstalledError`: the chosen `checkpoint` is not on this
  machine. Run `eluate setup` once.
- `PermissionError`: input not readable or output not writable.
- `EluateError`: any other eluate-specific failure.

See [Exceptions](#exceptions) for the full hierarchy.

---

<a id="class-eluate-session"></a>
### `class eluate.Session(*, output_dir=None, overwrite=False, force=False, audio_codec="aac", audio_bitrate="256k", on_progress=None, device=None, checkpoint="multi")`

Stateful session that amortises model loading across calls. The model
is loaded on the first `.elute()` call and reused for every subsequent
call within the session. Constructor kwargs become **session-level
defaults**; `Session.elute()` accepts the same kwargs as **per-call
overrides** that win when both are set.

`device` and `checkpoint` are session-locked: changing either would
require reloading the model, so they cannot be overridden per call.
Attempting to pass them to `Session.elute()` raises `TypeError`.

```python
import eluate

with eluate.Session(output_dir="out/", overwrite=True) as session:
    session.elute("a.mp4")                                 # uses session defaults
    session.elute("b.mp4", outputs=("speech",))            # different stems
    session.elute("c.mp4", output_dir="out/c/")            # per-call override
```

#### `Session.elute(input, *, outputs=("video",), output_dir=..., overwrite=..., force=..., audio_codec=..., audio_bitrate=..., on_progress=...) -> Result`

Run eluate on a single video and return a
[`Result`](#class-eluate-result). Per-call kwargs override the session
defaults; omitted kwargs fall back to whatever the `Session` was
constructed with. See [`eluate.elute`](#eluate-elute) above for
parameter semantics.

`device` and `checkpoint` are intentionally absent from this
signature; passing either raises `TypeError`.

**Returns:** [`Result`](#class-eluate-result).

**Raises:** same set as [`eluate.elute`](#eluate-elute), plus
`TypeError` if `device` or `checkpoint` is passed.

---

<a id="class-eluate-result"></a>
### `class eluate.Result`

Frozen dataclass returned by every successful run. A `None` on a stem
field means *the caller did not request that output*, never *a failure
occurred*; failures raise.

**Attributes**

- **input** (`pathlib.Path`): absolute path to the input video, after
  `~` expansion and resolution.
- **video** (`Optional[pathlib.Path]`): path to the cleaned video, or
  `None` if `"video"` was not in `outputs`.
- **speech** (`Optional[pathlib.Path]`): path to the speech-only WAV,
  or `None` if `"speech"` was not in `outputs`.
- **sfx** (`Optional[pathlib.Path]`): path to the SFX-only WAV, or
  `None` if `"sfx"` was not in `outputs`.
- **duration** (`float`): input video duration, in seconds.
- **processing_time** (`float`): wall-clock time the run took, in
  seconds. Useful for benchmarking.

```pycon
>>> result = eluate.elute("clip.mp4", outputs=("video", "speech"))
>>> result.video
PosixPath('/cwd/clip_eluted.mp4')
>>> result.speech
PosixPath('/cwd/clip_speech.wav')
>>> result.sfx is None
True
```

---

## Progress reporting

Pass any callable matching `Callable[[float, str], None]`:

```python
from tqdm import tqdm

bar = tqdm(total=100)
def on_progress(fraction: float, stage: str) -> None:
    bar.n = int(fraction * 100)
    bar.set_description(stage)
    bar.refresh()

eluate.elute("documentary.mp4", on_progress=on_progress)
```

Contract:

- `fraction` is monotonic non-decreasing across the
  `extract → load_model → separate → compile` pipeline.
- A final tick at `fraction == 1.0` is emitted on success, labelled with
  the last stage (`"compile"`).
- Internally each of the four stages is mapped to an equal quarter of the
  total range, so a UI sees smooth progress across stage boundaries.

---

<a id="exceptions"></a>
## Exceptions

Eluate errors form a small typed hierarchy under `eluate.EluateError`.
Standard library exceptions (`FileNotFoundError`, `PermissionError`,
`KeyboardInterrupt`) propagate **unwrapped**; eluate does not
re-package errors that already have the right Python name.

| Exception                   | Raised when                                                                 | Bypass |
|-----------------------------|------------------------------------------------------------------------------|--------|
| `EluateError`               | Base class. Catch this to handle any eluate-specific failure.                |        |
| `DurationOutOfRange`        | Input duration is zero, negative, or above the supported cap.                | `force=True` |
| `InsufficientDiskSpace`     | Target filesystem does not have enough free space.                           | `force=True` |
| `ModelNotInstalledError`    | Required checkpoint is not on this machine. The API never auto-downloads.    | Run `eluate setup` once. |
| `FileNotFoundError`         | Input video does not exist. *(stdlib, propagates unwrapped)*                 |        |
| `FileExistsError`           | An output path already exists and `overwrite=False`.                         | `overwrite=True` |
| `PermissionError`           | The pipeline cannot read input or write output. *(stdlib, propagates unwrapped)* |    |

```python
import eluate

try:
    eluate.elute("interview.mp4", outputs=("speech",))
except eluate.ModelNotInstalledError:
    print("Run `eluate setup` to download the checkpoint, then retry.")
except eluate.EluateError as exc:
    print(f"eluate failed: {exc}")
except FileNotFoundError as exc:
    print(f"input missing: {exc}")
```

The original low-level exception is chained via `__cause__`, so
debuggers and `logging.exception` see both layers.

---

## Logging

ELUATE routes all diagnostic output through stdlib `logging` under the
`eluate.api` and `eluate.core.*` loggers. The library installs no
handlers; by default nothing prints. Wire it up the standard way:

```python
import logging
logging.basicConfig(level=logging.INFO)
# now eluate progress and warnings appear on stderr
```

The CLI is a thin Rich-rendering layer over the same API and adds its
own console handler; importing `eluate` from your own code does not
inherit that.

---

## Stability and versioning

The four names listed at the top of this document plus their public
attributes are the public surface; everything else is internal.

ELUATE is pre-1.0 (`0.x`): the API is intended to be stable, but may
still change between releases until `1.0.0`. From `1.0.0` onward,
semantic versioning applies to this surface only:

- **MAJOR** bump for breaking changes to any of those names.
- **MINOR** bump for additive changes (new kwargs with defaults, new
  exception subclasses).
- **PATCH** bump for bug fixes and internal refactors.

Anything imported from `eluate.core`, `eluate.utils`, or `eluate.ui`
is internal; pin a version if you depend on it.
