# Benchmark in plain language

This is the short version. The full methodology, hardware details,
raw logs, and charts are in [`memory-benchmark.md`](memory-benchmark.md).

## The setup

Same model (Bandit v2), same checkpoint, same audio files, run
through two code paths:

1. The vendor's `demix()`, ZFTurbo's upstream implementation. It
   loads the whole track into memory.
2. ELUATE's `_demix_streaming`. It uses a small rolling buffer
   instead.

Three input lengths on an Apple M4 with 24 GB unified memory:

- short: 1.5 min
- medium: 14.5 min
- long: 84 min (a documentary)

## Memory: the main result

| Input | Vendor peak | ELUATE peak | Ratio |
|---|---:|---:|---:|
| short  | 13.5 GB | 13.6 GB | ~same |
| medium | 18.0 GB | 14.5 GB | 1.24× |
| long   | **41.2 GB** | 19.2 GB | **2.15×** |

The longer the input, the bigger ELUATE's advantage. On the 84-min
file the vendor used 41 GB on a 24 GB machine, and only finished
because macOS swapped ~20 GB to disk. ELUATE stayed under 20 GB
with no swap.

## Speed

Comparable within a few percent at every input length. The vendor's
wall-clock on the long input isn't a clean comparison anyway, because
it's contaminated by swap I/O cost while ELUATE's isn't.

## Output correctness

Before the fix, ELUATE's output diverged from the vendor's by
61–111 dB PSNR. The cause was a fade-in windowing bug in ELUATE's
port of the vendor's demix (not in the streaming itself). After
the fix and a re-run:

| Input | Before fix | After fix |
|---|---:|---:|
| short  | 61 dB  | **bit-identical** |
| medium | 84 dB  | 150+ dB |
| long   | 111 dB | 130+ dB |

130+ dB PSNR is essentially at the float32 precision limit. The
remaining tiny gap is MPS op-level non-determinism, not a code
difference.

## Caveats

- The short input is too small for ELUATE to show a memory
  advantage; its fixed rolling buffer is about the same size as
  the vendor's full-track buffer at 1.5 min.
- The vendor's long-run wall-time is contaminated by swap I/O
  cost, so it isn't a clean speed comparison.
- The vendor has the mirror bug (fade-out at end of last batch);
  ELUATE's copy of the function has the corresponding fix applied.

## Bottom line

ELUATE's streaming path is here to keep the tool usable on a long
documentary without needing 64 GB of RAM. The numbers say it does:

- 2.15× less peak memory on an 84-minute documentary.
- Comparable wall-clock speed (within a few percent).
- Bit-identical output at short inputs, near-float-precision
  parity at long inputs after the windowing fix.
