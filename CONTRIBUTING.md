# Contributing

This project was written to answer a question the documentation of this hardware does not
answer: *what does the MDLA actually do, and how fast, and where does it silently lie to
you?* Contributions are most useful when they extend that body of measurement.

## The most valuable thing you can send

**A measurement.** A number, the graph that produced it, the SoC and the Neuron version.
Particularly:

* A different SoC. Everything here is from one MT6897. A `-516` on your device that compiles
  fine here, or the reverse, is genuinely new information.
* A resolution of the **DIV / LOG / SIN conflict**. [operators.md §3](docs/operators.md#3-reported-as-unavailable)
  records two contradictory claims from different dates, and no log survives from the second.
  Run `测试/optable.py` and send the output.
* A CONV_TRANSPOSE result. It has never been measured on this hardware and it decides whether
  a whole class of vocoders is portable.
* A corrected number. If one of ours is wrong, say so with the reproduction.

## Reporting a problem

Minimum useful report:

```
1. SoC + NPU generation + Neuron version
2. the command you ran, verbatim
3. the complete output, not the line you think is the error
4. the graph text (.txt) if the failure is build/compile/run
5. for wrong numbers: the expected value and the observed value, and which layer
```

The graph format is plain text, so a bug report about a compile failure is usually a
20-line file. Attach it.

**Do not paste a screenshot of a terminal.** Copy the text.

## Before you open a pull request

* `git status` must be clean of artifacts. The `.gitignore` covers the common cases, but a
  stray `*.dla`, model checkpoint or vendor `.so` will make a PR unreviewable.
* If you change an operator's status or an error number, cite the measurement that changed
  it. The tables in this repository are evidence, not opinion — if the evidence is bad, the
  table should be corrected, not quietly patched.
* If you add a pitfall, describe the *symptom* first (what it looks like from outside) and
  then the cause. The symptom is what the next person will search for.

## Style

Comments in this repository are Chinese in the older modules and English in the newer ones.
**Do not "fix" that by mass-translating.** A lot of those comments record why a particular
dead end was abandoned; rewriting them destroys the provenance without adding information.
New code: either language is fine, matching the file it lives in.

Python targets 3.10+. Type hints are welcome but not required.

## What must not be committed

* `.dla` blobs, model checkpoints, `.so` files, APKs, keystores — the `.gitignore` blocks
  most of these; do not force-add them.
* Vendor headers or libraries. `NeuronAdapter.h` carries an explicit MediaTek
  confidential notice.
* Any PC-side tooling that patches the closed-source vendor compiler at runtime. Such patches
  are a local reverse-engineering aid and are deliberately outside this repository.

See [NOTICE](NOTICE) for the reasoning.

## License

Contributions are accepted under Apache-2.0. By opening a pull request you confirm you have
the right to license the contribution that way.
