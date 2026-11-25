from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt, welch

# One sample is 2 ms. Conditions are always shown as Baseline, then IFT, then Cut.
SAMPLING_RATE_HZ = 500
CONDITION_ORDER = {"Baseline": 0, "IFT": 1, "Cut": 2}

# Gap cutoffs for this project, not values taken from the papers.
# At 500 Hz, 50 samples is 100 ms and 250 samples is 500 ms.
GAP_SHORT_MAX = 50
GAP_MEDIUM_MAX = 250

# Ten-second analysis settings. Change them here to rerun the metrics.
EPOCH_SECONDS = 10
MAX_IMPUTED_FRACTION = 0.05
WELCH_NPERSEG = 1000
# A boundary belongs to the higher band. Gamma keeps 40 Hz, the filter edge.
FREQUENCY_BANDS = (
    ("Delta", 1, 4),
    ("Theta", 4, 8),
    ("Alpha", 8, 13),
    ("Beta", 13, 30),
    ("Gamma", 30, 40),
)
FANO_Z_THRESHOLD = 2.0
FANO_BIN_SECONDS = 0.008


def _ordered_recordings(dataset):
    """Recordings sorted by subject, then Baseline, IFT, Cut."""
    return sorted(
        dataset["recordings"].items(),
        key=lambda item: (item[1]["subject_id"], CONDITION_ORDER[item[1]["condition"]]),
    )


def _true_runs(mask):
    """Inclusive start and end of each consecutive True stretch."""
    edges = np.diff(np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return starts, ends


# Step 1. Load the CSV files and describe what is missing.


def recording_name(filename):
    """Read the subject and condition from a file name.

    014_baseline_filtered.csv is subject 014, Baseline.
    IFT stays IFT. A file named surgery is the Cut condition.
    """
    subject_id, stage, *_ = Path(filename).stem.split("_")
    condition = {"baseline": "Baseline", "ift": "IFT", "surgery": "Cut"}[stage.lower()]
    return subject_id, condition


def load_recordings(data_dir="data", sampling_rate_hz=SAMPLING_RATE_HZ):
    """Read every CSV in the data folder as it is stored.

    Missing samples stay NaN. Returns the sampling rate and one dictionary
    per recording, with the subject, condition, file name, and EEG table.
    """
    recordings = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        subject_id, condition = recording_name(path.name)
        recordings[f"{subject_id}_{condition}"] = {
            "subject_id": subject_id,
            "condition": condition,
            "source_file": path.name,
            "eeg": pd.read_csv(path),
        }
    return {"sampling_rate_hz": sampling_rate_hz, "recordings": recordings}


def recording_summary(dataset):
    """One row per recording: length, channels, and how much is missing.

    A channel with data has at least one sample. Completely missing channels
    are NaN from the first sample to the last, and they are included in the
    missing percentage.
    """
    rows = []
    for _, recording in _ordered_recordings(dataset):
        eeg = recording["eeg"]
        n_samples, n_channels = eeg.shape
        n_missing = int(eeg.isna().sum().sum())
        empty_channels = [column for column in eeg.columns if eeg[column].isna().all()]
        rows.append({
            "subject_id": recording["subject_id"],
            "condition": recording["condition"],
            "source_file": recording["source_file"],
            "n_samples": n_samples,
            "duration_seconds": n_samples / dataset["sampling_rate_hz"],
            "n_channels": n_channels,
            "n_channels_with_data": n_channels - len(empty_channels),
            "n_missing_values": n_missing,
            "missing_percent": 100.0 * n_missing / eeg.size,
            "completely_missing_channels": ", ".join(empty_channels),
        })
    return pd.DataFrame(rows)


def missing_percent_by_channel(dataset):
    """Missing percentage for every recording and channel. 100 means empty."""
    channel_names = sorted(
        {column for _, recording in _ordered_recordings(dataset) for column in recording["eeg"].columns},
        key=lambda name: int(name[2:]),
    )
    rows = {}
    for key, recording in _ordered_recordings(dataset):
        eeg = recording["eeg"]
        rows[key] = [100.0 * eeg[channel].isna().mean() for channel in channel_names]
    return pd.DataFrame.from_dict(rows, orient="index", columns=channel_names)


def plot_missing_percentages(dataset):
    """Heatmap of missing percentage by recording and channel."""
    table = missing_percent_by_channel(dataset)
    values = table.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=100, cmap="YlOrRd")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("EEG channel")
    ax.set_ylabel("Recording")
    ax.set_title("Missing EEG values by recording and channel (%)")
    fig.colorbar(image, ax=ax).set_label("Missing values (%)")

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if value == 0:
                label = "0"
            else:
                label = f"{value:.1f}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 50 else "black",
            )

    fig.tight_layout()
    return fig


# Step 2. Label each missing stretch by its length.


def classify_missing_gaps(dataset):
    """Find each run of missing samples and label it by length.

    One sample is Single, 2–50 is Short, 51–250 is Medium, and a longer
    partial gap is Long. A channel missing from start to end is Complete
    and is counted separately, not as an ordinary gap.
    Returns a summary, one row per gap, and a stacked bar of ordinary gaps.
    """
    gap_rows = []
    summary_rows = []
    for _, recording in _ordered_recordings(dataset):
        eeg = recording["eeg"]
        n_samples = len(eeg)
        ordinary = {"Single": 0, "Short": 0, "Medium": 0, "Long": 0}
        n_complete = 0
        n_missing = 0
        longest_partial_ms = 0.0

        for channel in eeg.columns:
            missing = eeg[channel].isna().to_numpy()
            if not missing.any():
                continue
            for start, end in zip(*_true_runs(missing)):
                length = int(end - start + 1)
                duration_ms = length * 1000.0 / dataset["sampling_rate_hz"]
                if length == n_samples:
                    category = "Complete"
                    n_complete += 1
                elif length == 1:
                    category = "Single"
                elif length <= GAP_SHORT_MAX:
                    category = "Short"
                elif length <= GAP_MEDIUM_MAX:
                    category = "Medium"
                else:
                    category = "Long"

                n_missing += length
                if category != "Complete":
                    ordinary[category] += 1
                    longest_partial_ms = max(longest_partial_ms, duration_ms)

                gap_rows.append({
                    "subject_id": recording["subject_id"],
                    "condition": recording["condition"],
                    "channel": channel,
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "gap_length_samples": length,
                    "gap_duration_ms": duration_ms,
                    "gap_category": category,
                })

        summary_rows.append({
            "subject_id": recording["subject_id"],
            "condition": recording["condition"],
            "n_gaps": sum(ordinary.values()),
            "n_single": ordinary["Single"],
            "n_short": ordinary["Short"],
            "n_medium": ordinary["Medium"],
            "n_long": ordinary["Long"],
            "n_complete_channels": n_complete,
            "n_missing_samples": n_missing,
            "missing_percent": 100.0 * n_missing / eeg.size,
            "longest_partial_gap_ms": longest_partial_ms,
        })

    gap_summary = pd.DataFrame(summary_rows)
    gap_details = pd.DataFrame(gap_rows)
    labels = [f"{row.subject_id} {row.condition}" for row in gap_summary.itertuples(index=False)]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(gap_summary))
    for category, color in zip(("Single", "Short", "Medium", "Long"), colors):
        counts = gap_summary[f"n_{category.lower()}"].to_numpy()
        ax.bar(labels, counts, bottom=bottom, color=color, label=category, width=0.7)
        bottom = bottom + counts
    ax.set_xlabel("Recording")
    ax.set_ylabel("Number of gaps")
    ax.set_title("Missing EEG gaps by category")
    ax.legend(title="Gap category")
    fig.tight_layout()
    return gap_summary, gap_details, fig


# Step 3. Fill short gaps. Leave long gaps and empty channels missing.


def impute_recording(eeg, gaps):
    """Fill eligible gaps in one recording. The original table is not changed.

    One missing sample is the straight line between the nearest observed
    samples. Short gaps (2–50 samples) use PCHIP. Medium gaps (51–250)
    also use PCHIP, but only as a provisional fill. Longer gaps and a
    completely missing channel stay NaN. Nothing is extrapolated past the
    first or last observed sample.
    Returns the filled copy, a row for every gap, and the fill counts.
    """
    filled = eeg.copy()
    gaps = gaps if gaps is not None else pd.DataFrame()
    subject_id = gaps["subject_id"].iloc[0] if len(gaps) else ""
    condition = gaps["condition"].iloc[0] if len(gaps) else ""
    log_rows = []
    n_linear = 0
    n_pchip = 0
    n_skipped = 0
    n_provisional = 0

    for channel in eeg.columns:
        channel_gaps = gaps[gaps["channel"] == channel] if len(gaps) else gaps
        if len(channel_gaps) == 0:
            continue

        observed = eeg[channel].to_numpy(dtype=float)
        values = observed.copy()
        valid_index = np.flatnonzero(~np.isnan(observed))
        # PCHIP is fit only on observed samples, never on a value just filled.
        pchip = None
        if len(valid_index) >= 2:
            pchip = PchipInterpolator(valid_index, observed[valid_index], extrapolate=False)

        for gap in channel_gaps.itertuples(index=False):
            start = int(gap.start_sample)
            end = int(gap.end_sample)
            category = gap.gap_category
            positions = np.arange(start, end + 1)
            positions = positions[np.isnan(observed[positions])]
            has_both_sides = (
                len(valid_index) >= 2 and valid_index[0] < start and valid_index[-1] > end
            )
            action = "skipped"
            n_filled = 0

            if category == "Long":
                reason = "long gap left missing"
            elif category == "Complete":
                reason = "complete channel left missing"
            elif len(positions) == 0:
                reason = "gap was not missing in the original signal"
            elif not has_both_sides:
                reason = "no observed sample on both sides"
            elif category == "Single":
                left = valid_index[valid_index < start][-1]
                right = valid_index[valid_index > end][0]
                for index in positions:
                    weight = (index - left) / (right - left)
                    values[index] = (1 - weight) * observed[left] + weight * observed[right]
                n_filled = len(positions)
                n_linear += n_filled
                action = "linear"
                reason = "filled from the nearest observed samples"
            elif category in ("Short", "Medium"):
                estimated = np.asarray(pchip(positions), dtype=float)
                if not np.all(np.isfinite(estimated)):
                    reason = "interpolation was not finite"
                else:
                    values[positions] = estimated
                    n_filled = len(positions)
                    n_pchip += n_filled
                    if category == "Medium":
                        action = "provisional_pchip"
                        n_provisional += 1
                        reason = "provisional PCHIP baseline"
                    else:
                        action = "pchip"
                        reason = "filled with PCHIP"
            else:
                reason = "category was not recognized"

            if action == "skipped":
                n_skipped += 1
            log_rows.append({
                "subject_id": subject_id,
                "condition": condition,
                "channel": channel,
                "start_sample": start,
                "end_sample": end,
                "gap_category": category,
                "action": action,
                "n_samples_filled": n_filled,
                "reason": reason,
            })

        filled[channel] = values

    summary = {
        "subject_id": subject_id,
        "condition": condition,
        "n_missing_original": int(eeg.isna().sum().sum()),
        "n_linear_samples": n_linear,
        "n_pchip_samples": n_pchip,
        "n_missing_remaining": int(filled.isna().sum().sum()),
        "n_skipped_gaps": n_skipped,
        "n_provisional_gaps": n_provisional,
    }
    return filled, pd.DataFrame(log_rows), summary


def plot_imputation_example(original, imputed, gap_log):
    """Show one filled short gap and one provisional medium gap.

    A smooth line only shows what the interpolant did. It does not show
    that the missing EEG was recovered.
    """
    short_rows = gap_log[gap_log["action"] == "pchip"]
    medium_rows = gap_log[gap_log["action"] == "provisional_pchip"]
    if short_rows.empty or medium_rows.empty:
        raise ValueError("The gap log needs one filled short gap and one provisional medium gap.")

    shared_channels = sorted(
        set(short_rows["channel"]) & set(medium_rows["channel"]),
        key=lambda name: int(name[2:]),
    )
    if shared_channels:
        channel = shared_channels[0]
        short_gap = short_rows[short_rows["channel"] == channel].iloc[0]
        medium_gap = medium_rows[medium_rows["channel"] == channel].iloc[0]
    else:
        short_gap = short_rows.iloc[0]
        medium_gap = medium_rows.iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for axis, gap, title in (
        (axes[0], short_gap, "Short gap, PCHIP"),
        (axes[1], medium_gap, "Medium gap, provisional PCHIP"),
    ):
        channel = gap["channel"]
        start = int(gap["start_sample"])
        end = int(gap["end_sample"])
        context = max(80, end - start + 1)
        left = max(0, start - context)
        right = min(len(original) - 1, end + context)
        samples = np.arange(left, right + 1)
        axis.plot(samples, original[channel].iloc[left:right + 1], color="black", linewidth=1, label="Original")
        axis.plot(
            np.arange(start, end + 1),
            imputed[channel].iloc[start:end + 1],
            color="#E45756",
            linewidth=1.5,
            label="Reconstructed",
        )
        axis.axvspan(start, end, color="#E45756", alpha=0.15, label="Filled samples")
        axis.set_title(f"{gap['subject_id']} {gap['condition']}, {channel}: {title}")
        axis.set_xlabel("Sample")
        axis.set_ylabel("EEG")
        axis.legend(loc="upper right")

    fig.tight_layout()
    return fig


# Step 4. Band-pass filter from 1 to 40 Hz, without crossing a gap.


def bandpass_recording(eeg, sampling_rate_hz=SAMPLING_RATE_HZ):
    """Apply a 4th-order Butterworth band-pass of 1–40 Hz to each channel.

    Filtering runs forward and backward on each continuous finite stretch,
    so it does not cross a NaN gap or replace a NaN with zero. A stretch
    shorter than the filter padding is left as it is and counted.
    The filter attenuates frequencies outside 1–40 Hz. It does not delete samples.
    """
    sos = butter(4, [1, 40], btype="bandpass", fs=sampling_rate_hz, output="sos")
    # Same pad length sosfiltfilt uses. A shorter run cannot be filtered without inventing edges.
    n_sections = sos.shape[0]
    pad_length = 3 * (
        2 * n_sections + 1 - min(int(np.sum(sos[:, 2] == 0)), int(np.sum(sos[:, 5] == 0)))
    )

    filtered = eeg.copy()
    n_filtered = 0
    n_too_short = 0
    for channel in eeg.columns:
        values = eeg[channel].to_numpy(dtype=float)
        output = values.copy()
        for start, end in zip(*_true_runs(np.isfinite(values))):
            if end - start + 1 <= pad_length:
                n_too_short += 1
                continue
            output[start:end + 1] = sosfiltfilt(sos, values[start:end + 1])
            n_filtered += 1
        filtered[channel] = output

    return filtered, {"n_segments_filtered": n_filtered, "n_segments_too_short": n_too_short}


# Step 5. Keep 10-second epochs, then measure spectrum, complexity, and events.


def prepare_recording(
    original,
    imputed,
    filtered,
    subject_id,
    condition,
    sampling_rate_hz=SAMPLING_RATE_HZ,
    epoch_seconds=EPOCH_SECONDS,
    max_imputed_fraction=MAX_IMPUTED_FRACTION,
):
    """Decide which 10-second blocks of one recording can enter the metrics.

    Channels missing for the whole file are dropped. Epochs are cut only
    inside a stretch where every remaining channel is finite, and stretches
    are not joined across a NaN gap. A kept channel may contain at most 5%
    imputed samples. Imputed means NaN in the original file and finite after Step 3.
    READY means every full block passed, PARTIAL means some were dropped,
    and NOT READY means none passed.
    """
    n_samples = len(original)
    epoch_length = int(epoch_seconds * sampling_rate_hz)
    kept = [channel for channel in original.columns if original[channel].notna().any()]
    excluded = [channel for channel in original.columns if channel not in kept]
    n_cells = original.size
    report = {
        "subject_id": subject_id,
        "condition": condition,
        "duration_seconds": n_samples / sampling_rate_hz,
        "channels_used": ", ".join(kept),
        "channels_excluded": ", ".join(excluded) if excluded else "None",
        "original_missing_percent": 100 * original.isna().sum().sum() / n_cells,
        "imputed_percent": 100 * (original.isna() & imputed.notna()).sum().sum() / n_cells,
        "remaining_missing_percent": 100 * filtered.isna().sum().sum() / filtered.size,
        "n_usable_epochs": 0,
        "n_excluded_epochs": 0,
        "exclusion_reason": "None",
        "status": "NOT READY",
    }
    if not kept or epoch_length < 1:
        report["exclusion_reason"] = "No channel has any recorded samples"
        report["n_excluded_epochs"] = n_samples // epoch_length if epoch_length else 0
        return report, []

    imputed_flags = {
        channel: (original[channel].isna() & imputed[channel].notna()).to_numpy()
        for channel in kept
    }
    epochs = []
    n_short_runs = 0
    n_failed_epochs = 0
    for start, end in zip(*_true_runs(np.isfinite(filtered[kept].to_numpy(dtype=float)).all(axis=1))):
        if end - start + 1 < epoch_length:
            n_short_runs += 1
            continue
        for block in range((end - start + 1) // epoch_length):
            first = start + block * epoch_length
            last = first + epoch_length
            too_imputed = any(
                imputed_flags[channel][first:last].mean() > max_imputed_fraction for channel in kept
            )
            if too_imputed:
                n_failed_epochs += 1
                continue
            epochs.append({
                "subject_id": subject_id,
                "condition": condition,
                "start_sample": int(first),
                "data": filtered.iloc[first:last][kept].reset_index(drop=True),
            })

    reasons = []
    if n_failed_epochs:
        reasons.append(
            f"{n_failed_epochs} epochs had more than {max_imputed_fraction:.0%} imputed samples in a channel"
        )
    if n_short_runs:
        reasons.append(f"{n_short_runs} finite stretches were shorter than {epoch_seconds:g} seconds")
    report["n_usable_epochs"] = len(epochs)
    report["n_excluded_epochs"] = n_failed_epochs + n_short_runs
    report["exclusion_reason"] = "; ".join(reasons) if reasons else "None"
    if epochs and report["n_excluded_epochs"] == 0:
        report["status"] = "READY"
    elif epochs:
        report["status"] = "PARTIAL"
    return report, epochs


def welch_band_power(epochs, sampling_rate_hz=SAMPLING_RATE_HZ, nperseg=WELCH_NPERSEG):
    """Welch power in delta, theta, alpha, beta, and gamma for each epoch.

    The segment length is 1000 samples, so the bins are 0.5 Hz. Overlap is
    SciPy's default, half of that segment. Absolute power is the area of a
    band. Relative power divides by the sum of the five areas. A shared
    boundary goes to the higher band, and 40 Hz stays in gamma.
    """
    band_rows = []
    spectrum_rows = []
    for epoch in epochs:
        for channel in epoch["data"].columns:
            values = epoch["data"][channel].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                continue
            frequencies, power = welch(values, fs=sampling_rate_hz, nperseg=nperseg)
            in_range = (frequencies >= 1) & (frequencies <= 40)
            for frequency, value in zip(frequencies[in_range], power[in_range]):
                spectrum_rows.append({
                    "subject_id": epoch["subject_id"],
                    "condition": epoch["condition"],
                    "start_sample": epoch["start_sample"],
                    "channel": channel,
                    "frequency": float(frequency),
                    "power": float(value),
                })
            absolute = {}
            for name, low, high in FREQUENCY_BANDS:
                if name == "Gamma":
                    selected = (frequencies >= low) & (frequencies <= high)
                else:
                    selected = (frequencies >= low) & (frequencies < high)
                absolute[name] = float(np.trapezoid(power[selected], frequencies[selected]))
            total = sum(absolute.values())
            for name, value in absolute.items():
                band_rows.append({
                    "subject_id": epoch["subject_id"],
                    "condition": epoch["condition"],
                    "start_sample": epoch["start_sample"],
                    "channel": channel,
                    "band": name,
                    "absolute_power": value,
                    "relative_power": value / total if total > 0 else np.nan,
                })
    return pd.DataFrame(band_rows), pd.DataFrame(spectrum_rows)


def spectral_exponent(spectra):
    """Slope of the epoch-averaged spectrum, fit with specparam.

    Fixed aperiodic mode is fit on 1–40 Hz, and again on 1–20 Hz and
    20–40 Hz. A larger positive exponent is a steeper fall in log-log space.
    This is not a consciousness measure. The 1 Hz and 40 Hz filter edges can
    bend the fit, and 20 Hz is included in both split ranges.
    """
    from specparam import SpectralModel

    empty_fit = pd.DataFrame(columns=[
        "subject_id", "condition", "channel", "frequency_range", "exponent", "r_squared",
    ])
    empty_curve = pd.DataFrame(columns=[
        "subject_id", "condition", "channel", "frequency", "mean_power", "aperiodic_power",
    ])
    if spectra.empty:
        return empty_fit, empty_curve

    averaged = (
        spectra.groupby(["subject_id", "condition", "channel", "frequency"], as_index=False)["power"]
        .mean()
        .rename(columns={"power": "mean_power"})
    )
    fit_rows = []
    curve_rows = []
    ranges = (("1-40 Hz", 1, 40), ("1-20 Hz", 1, 20), ("20-40 Hz", 20, 40))
    for (subject_id, condition, channel), channel_spectrum in averaged.groupby(
        ["subject_id", "condition", "channel"]
    ):
        frequencies = channel_spectrum["frequency"].to_numpy(dtype=float)
        power = channel_spectrum["mean_power"].to_numpy(dtype=float)
        for label, low, high in ranges:
            selected = (frequencies >= low) & (frequencies <= high)
            exponent = np.nan
            r_squared = np.nan
            offset = np.nan
            if selected.sum() >= 4 and np.all(power[selected] > 0):
                model = SpectralModel(
                    aperiodic_mode="fixed",
                    verbose=False,
                    peak_width_limits=[1, 6],
                    min_peak_height=0.05,
                    max_n_peaks=6,
                )
                try:
                    model.fit(frequencies[selected], power[selected], freq_range=[low, high])
                    if model.results.has_model:
                        exponent = float(model.get_params("aperiodic", "exponent"))
                        offset = float(model.get_params("aperiodic", "offset"))
                        r_squared = float(model.to_df()["gof_rsquared"])
                except Exception:
                    exponent = np.nan
            fit_rows.append({
                "subject_id": subject_id,
                "condition": condition,
                "channel": channel,
                "frequency_range": label,
                "exponent": exponent,
                "r_squared": r_squared,
            })
            if label == "1-40 Hz" and np.isfinite(exponent):
                fitted = 10 ** (offset - exponent * np.log10(frequencies[selected]))
                for frequency, observed, predicted in zip(frequencies[selected], power[selected], fitted):
                    curve_rows.append({
                        "subject_id": subject_id,
                        "condition": condition,
                        "channel": channel,
                        "frequency": float(frequency),
                        "mean_power": float(observed),
                        "aperiodic_power": float(predicted),
                    })
    return pd.DataFrame(fit_rows), pd.DataFrame(curve_rows)


def lempel_ziv_complexity(epochs):
    """Normalized Lempel-Ziv complexity of each channel and epoch.

    Samples above that epoch's own mean become 1 and the rest become 0.
    antropy divides the complexity by n / log2(n). This is not PCI.
    """
    from antropy import lziv_complexity

    rows = []
    for epoch in epochs:
        for channel in epoch["data"].columns:
            values = epoch["data"][channel].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                continue
            binary = (values > values.mean()).astype(int)
            rows.append({
                "subject_id": epoch["subject_id"],
                "condition": epoch["condition"],
                "start_sample": epoch["start_sample"],
                "channel": channel,
                "lzc": float(lziv_complexity(binary, normalize=True)),
            })
    return pd.DataFrame(rows)


def fano_factor(
    epochs,
    sampling_rate_hz=SAMPLING_RATE_HZ,
    threshold=FANO_Z_THRESHOLD,
    bin_seconds=FANO_BIN_SECONDS,
):
    """Fano factor of event counts, not of the raw EEG samples.

    Each channel is z-scored inside the epoch. An event starts when the
    absolute z-score crosses 2. Later samples in the same excursion are not
    counted again. Starts are summed across channels in 8 ms bins, and the
    factor is the variance of those counts divided by their mean. It is NaN
    when an epoch has no events. The threshold and the absolute-value rule
    are project choices.
    """
    bin_size = max(1, int(round(bin_seconds * sampling_rate_hz)))
    summary_rows = []
    count_rows = []
    for epoch in epochs:
        channels = [
            epoch["data"][channel].to_numpy(dtype=float)
            for channel in epoch["data"].columns
            if np.isfinite(epoch["data"][channel].to_numpy(dtype=float)).all()
        ]
        if not channels:
            continue
        n_bins = len(channels[0]) // bin_size
        counts = np.zeros(n_bins)
        for values in channels:
            centered = values - values.mean()
            scale = centered.std()
            if scale == 0:
                continue
            above = np.abs(centered / scale) >= threshold
            starts = above.copy()
            starts[1:] &= ~above[:-1]
            counts += starts[:n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)
        mean_count = float(counts.mean())
        factor = np.nan if mean_count == 0 else float(counts.var() / mean_count)
        summary_rows.append({
            "subject_id": epoch["subject_id"],
            "condition": epoch["condition"],
            "start_sample": epoch["start_sample"],
            "mean_event_count": mean_count,
            "fano_factor": factor,
        })
        for index, count in enumerate(counts):
            count_rows.append({
                "subject_id": epoch["subject_id"],
                "condition": epoch["condition"],
                "start_sample": epoch["start_sample"],
                "time_seconds": index * bin_size / sampling_rate_hz,
                "event_count": int(count),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(count_rows)
