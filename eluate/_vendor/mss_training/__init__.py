# SPDX-License-Identifier: MIT
"""
Frozen subset of ZFTurbo's Music-Source-Separation-Training framework.

Original repository:
    https://github.com/ZFTurbo/Music-Source-Separation-Training

Eluate fork (upstream-tracking submodule):
    https://github.com/borderedprominent/mss-training-eluate

Only the modules eluate invokes at inference time are bundled here:
``models.bandit_v2`` (the Bandit v2 model implementation) and a slim
subset of ``utils.settings`` (``load_config`` + the bandit_v2 branch of
``get_model_from_config``). The original framework also provides
training, validation, and many other model architectures — those are
not shipped with eluate.

Licensed under the MIT License; see the bundled ``LICENSE`` file in
this directory for the upstream copyright notice.
"""
