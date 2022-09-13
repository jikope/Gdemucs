# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import logging

import os
from pathlib import Path
from dora.log import LogProgress
import numpy as np
import musdb
import museval
import torch as th
import torchaudio as ta

from .apply import apply_model
from .audio import convert_audio, save_audio
from . import distrib
from .utils import DummyPoolExecutor


logger = logging.getLogger(__name__)


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores


def eval_track(references, estimates, win, hop, compute_sdr=True):
    references = references.transpose(1, 2).double()
    estimates = estimates.transpose(1, 2).double()

    new_scores = new_sdr(references.cpu()[None], estimates.cpu()[None])[0]

    if not compute_sdr:
        return None, new_scores
    else:
        references = references.numpy()
        estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]
        return scores, new_scores


def evaluate(solver, compute_sdr=False):
    """
    Evaluate model using museval.
    `new_only` means using only the MDX definition of the SDR, which is much faster to evaluate.
    """

    args = solver.args

    output_dir = solver.folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = solver.folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    src_rate = args.dset.samplerate

    # Load custom set
    # TEST_FOLDER = args.dset.wav + 'test'
    test_path = Path(args.dset.wav) / 'test'

    test_set = []

    for root, folders, files in os.walk(test_path, followlinks=True):
        root = Path(root)
        if root.name.startswith('.') or folders or root == test_path:
            continue
        name = str(root.relative_to(test_path))
        test_set.append(name)    
        
    eval_device = 'cpu'

    model = solver.model
    win = int(1. * model.samplerate)
    hop = int(1. * model.samplerate)

    indexes = range(distrib.rank, len(test_set), distrib.world_size)
    indexes = LogProgress(logger, indexes, updates=args.misc.num_prints,
                          name='Eval')
    pendings = []

    pool = futures.ProcessPoolExecutor if args.test.workers else DummyPoolExecutor
    with pool(args.test.workers) as pool:
        for name in test_set:
            mix, _ = ta.load(test_path / name / 'MIXTURE.wav')
            mix = mix.to(solver.device)
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(model, mix[None],
                                    shifts=args.test.shifts, split=args.test.split,
                                    overlap=args.test.overlap)[0]
            estimates = estimates * ref.std() + ref.mean()
            estimates = estimates.to(eval_device)

            references = th.stack(
                [ta.load(test_path / name / (source + '.wav'))[0] for source in model.sources])
            if references.dim() == 2:
                references = references[:, None]
                references = references.to(eval_device)
                references = convert_audio(references, src_rate,
                                           model.samplerate, model.audio_channels)
            if args.test.save:
                folder = solver.folder / "wav" / name
                folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(model.sources, estimates):
                    save_audio(estimate.cpu(), folder / (name + ".mp3"), model.samplerate)

            pendings.append((name, pool.submit(
                eval_track, references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)))

        pendings = LogProgress(logger, pendings, updates=args.misc.num_prints,
                               name='Eval (BSS)')
        tracks = {}
        for track_name, pending in pendings:
            pending = pending.result()
            scores, nsdrs = pending
            tracks[track_name] = {}
            for idx, target in enumerate(model.sources):
                tracks[track_name][target] = {'nsdr': [float(nsdrs[idx])]}
            if scores is not None:
                (sdr, isr, sir, sar) = scores
                for idx, target in enumerate(model.sources):
                    values = {
                        "SDR": sdr[idx].tolist(),
                        "SIR": sir[idx].tolist(),
                        "ISR": isr[idx].tolist(),
                        "SAR": sar[idx].tolist()
                    }
                    tracks[track_name][target].update(values)

        all_tracks = {}
        for src in range(distrib.world_size):
            all_tracks.update(distrib.share(tracks, src))

        result = {}
        metric_names = next(iter(all_tracks.values()))[model.sources[0]]
        for metric_name in metric_names:
            avg = 0
            avg_of_medians = 0
            for source in model.sources:
                medians = [
                    np.nanmedian(all_tracks[track][source][metric_name])
                    for track in all_tracks.keys()]
                mean = np.mean(medians)
                median = np.median(medians)
                result[metric_name.lower() + "_" + source] = mean
                result[metric_name.lower() + "_med" + "_" + source] = median
                avg += mean / len(model.sources)
                avg_of_medians += median / len(model.sources)
                result[metric_name.lower()] = avg
                result[metric_name.lower() + "_med"] = avg_of_medians
        return result
