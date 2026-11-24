

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt, welch

# region PROJECT SETTINGS
# EEG sampling frequency: 500 samples per second, so one sample is 2 ms.
SAMPLING_RATE_HZ = 500
# Display conditions in the same sequence throughout the study.
CONDITION_ORDER = {"Baseline": 0, "IFT": 1, "Cut": 2}

# Gap-length cutoffs for this project, not values taken from the papers.
# At 500 Hz, 50 samples is 100 ms and 250 samples is 500 ms.
GAP_SHORT_MAX = 50
GAP_MEDIUM_MAX = 250
# endregion


# region STEP 1 - LOAD AND INSPECT EEG RECORDINGS
# Step 1A - File identification and data loading

# WHAT: Extract the subject ID and recording stage from a CSV filename.
# WHY: Later analyses need consistent labels such as "014" and "Baseline".
# INPUT: A filename, e.g. "014_baseline (1).csv".
# OUTPUT: A pair (subject_id, condition), e.g. ("014", "Baseline").
def recording_name(filename):
    """Read the subject and condition from a CSV file name.

    Example: 014_baseline_filtered.csv is subject 014, Baseline.
    IFT stays IFT. A file named surgery is the Cut condition, the first
    surgical incision. A copy suffix such as " (1)" is ignored.

    Returns the subject id and the condition name.
    """
    stem = Path(filename).stem
    if " (" in stem and stem.endswith(")"):
        base, suffix = stem.rsplit(" (", 1)
        if suffix[:-1].isdigit():
            stem = base

    parts = stem.split("_", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        raise ValueError(f"Cannot read subject and condition from filename: {filename}")

    subject_id, rest = parts
    text = rest.lower()
    if "baseline" in text:
        condition = "Baseline"
    elif "ift" in text:
        condition = "IFT"
    elif "surgery" in text or "cut" in text:
        condition = "Cut"
    else:
        raise ValueError(f"Cannot read the recording condition from filename: {filename}")

    return subject_id, condition



# WHAT: Read every CSV in the data folder and identify EEG columns (Ch1, Ch2, ...).
# WHY: All later analysis steps should use the same organized dataset.
# INPUT: The data directory and EEG sampling frequency.
# OUTPUT: A dictionary containing the sampling rate and all recordings;
#         original NaN values and non-EEG metadata columns are preserved.
def load_recordings(data_dir="data", sampling_rate_hz=SAMPLING_RATE_HZ):
    """Read every EEG CSV in the data folder.

    A column named Ch plus a number (Ch1, Ch2, ...) is an EEG channel.
    The number of channels is taken from the file header. Any other column
    is kept aside as metadata. Missing values stay NaN. If a channel contains
    text that is not a number, those cells are counted and left unchanged.

    Returns the sampling rate and one dictionary per recording. Each
    recording has the subject, condition, source file, EEG table, any
    non-channel columns, and the count of non-numeric cells.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    # Search only for CSV files and keep the order reproducible.
    csv_paths = sorted(data_path.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    recordings = {}
    for path in csv_paths:
        subject_id, condition = recording_name(path.name)
        key = f"{subject_id}_{condition}"
        if key in recordings:
            raise ValueError(
                f"More than one file maps to subject {subject_id}, {condition}: {path.name}"
            )

        table = pd.read_csv(path)
        channel_names = []
        other_names = []
        for column in table.columns:
            text = str(column).strip()
            if len(text) > 2 and text[:2].lower() == "ch" and text[2:].isdigit():
                channel_names.append(column)
            else:
                other_names.append(column)
        if not channel_names:
            raise ValueError(f"No EEG channel columns found in {path.name}")

# Work on a copy of EEG columns, keeping non-EEG columns separate.
        eeg = table[channel_names].copy()
        n_invalid = 0
        for column in eeg.columns:
            series = eeg[column]
            if pd.api.types.is_numeric_dtype(series):
                eeg[column] = series.astype(float)
                continue
            # Keep real text. Do not turn it into NaN.
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = series.notna() & numeric.isna()
            n_invalid += int(invalid.sum())
            if not invalid.any():
                eeg[column] = numeric

        recordings[key] = {
            "subject_id": subject_id,
            "condition": condition,
            "source_file": path.name,
            "eeg": eeg,
            "other": table[other_names].copy() if other_names else pd.DataFrame(index=table.index),
            "n_invalid_values": n_invalid,
        }

    return {
        "sampling_rate_hz": sampling_rate_hz,
        "recordings": recordings,
    }


# Step 1B - Descriptive quality checks

# WHAT: Create one summary row for each EEG recording.
# WHY: Check recording duration, channel availability, and overall missingness.
# INPUT: The dataset returned by load_recordings().
# OUTPUT: A pandas DataFrame with counts, percentages, and empty channels.
# NOTE: Missing percentages include fully empty channels in the denominator.
def recording_summary(dataset):
    """Build one row per recording.

    The row has the subject, condition, file name, number of samples,
    duration in seconds, number of EEG channels, how many of those channels
    contain at least one value, the missing-value count and percentage,
    the channels that are entirely empty, and the count of non-numeric cells.

    Recordings are ordered by subject, then Baseline, IFT, Cut.
    """
    sampling_rate_hz = dataset["sampling_rate_hz"]
    ordered = sorted(
        dataset["recordings"].items(),
        key=lambda item: (
            item[1]["subject_id"],
            CONDITION_ORDER.get(item[1]["condition"], len(CONDITION_ORDER)),
            item[0],
        ),
    )

    rows = []
    for _, recording in ordered:
        eeg = recording["eeg"]
        n_samples = len(eeg)
        n_channels = eeg.shape[1]
# isna() marks NaN cells; the two sums count them across all channels.
        n_missing = int(eeg.isna().sum().sum())
        n_values = n_samples * n_channels
        empty_channels = [column for column in eeg.columns if eeg[column].isna().all()]

        rows.append(
            {
                "subject_id": recording["subject_id"],
                "condition": recording["condition"],
                "source_file": recording["source_file"],
                "n_samples": n_samples,
                "duration_seconds": n_samples / sampling_rate_hz,
                "n_channels": n_channels,
                "n_channels_with_data": n_channels - len(empty_channels),
                "n_missing_values": n_missing,
                "missing_percent": (100.0 * n_missing / n_values) if n_values else np.nan,
                "completely_missing_channels": ", ".join(empty_channels),
                "n_invalid_values": recording["n_invalid_values"],
            }
        )
    return pd.DataFrame(rows)



# WHAT: Calculate the percentage of NaN values for each EEG channel.
# WHY: A recording's overall missing percentage can hide problematic channels.
# INPUT: The loaded dataset.
# OUTPUT: A table: rows are recordings, columns are channels, values are %.
def missing_percent_by_channel(dataset):
    """Missing-value percentage for every recording and EEG channel.

    Each row is one recording. Each column is one channel, in order
    Ch1, Ch2, and so on. A value of 100 means that channel is empty.
    """
    channel_names = sorted(
        {column for recording in dataset["recordings"].values() for column in recording["eeg"].columns},
        key=lambda name: int(str(name)[2:]),
    )
    ordered = sorted(
        dataset["recordings"].items(),
        key=lambda item: (
            item[1]["subject_id"],
            CONDITION_ORDER.get(item[1]["condition"], len(CONDITION_ORDER)),
            item[0],
        ),
    )

    rows = {}
    for key, recording in ordered:
        eeg = recording["eeg"]
        n_samples = len(eeg)
        percentages = []
        for channel in channel_names:
            if channel not in eeg.columns or n_samples == 0:
                percentages.append(np.nan)
            else:
                percentages.append(100.0 * eeg[channel].isna().mean())
        rows[key] = percentages
    return pd.DataFrame.from_dict(rows, orient="index", columns=channel_names)



# WHAT: Display channel-by-channel missing percentages in a heatmap.
# WHY: Colors make channels with many missing samples easy to identify.
# INPUT: The loaded dataset.
# OUTPUT: A Matplotlib figure (does not modify EEG data).
def plot_missing_percentages(dataset):
    """Draw one heatmap of missing-value percentage by recording and channel.

    Returns the matplotlib figure. Darker cells have more missing values.
    """
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
            if np.isnan(value):
                label = ""
            elif value == 0:
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


# endregion

# region STEP 2 - CLASSIFY MISSING-DATA GAPS

# WHAT: Locate each consecutive run of NaN samples and classify its length.
# WHY: Short, medium, long, and fully missing segments need different handling.
# INPUT: The loaded dataset (500 Hz sampling rate by default).
# OUTPUT: (summary table, detailed gap table, stacked bar-chart figure).
# NOTE: The 1/50/250-sample thresholds are our project choices, not paper rules.
def classify_missing_gaps(dataset):
    sampling_rate_hz = dataset["sampling_rate_hz"]
    ordered = sorted(
        dataset["recordings"].items(),
        key=lambda item: (
            item[1]["subject_id"],
            CONDITION_ORDER.get(item[1]["condition"], len(CONDITION_ORDER)),
            item[0],
        ),
    )

    gap_rows = []
    summary_rows = []
    for _, recording in ordered:
        eeg = recording["eeg"]
        n_samples = len(eeg)
        n_channels = eeg.shape[1]
        ordinary = {"Single": 0, "Short": 0, "Medium": 0, "Long": 0}
        n_complete = 0
        n_missing = 0
        longest_partial_ms = 0.0

        for channel in eeg.columns:
            missing = eeg[channel].isna().to_numpy()
            if not missing.any():
                continue

            # A change from present to missing starts a gap. The next change ends it.
# Padding lets np.diff detect gaps at the very start or end of a recording.
            padded = np.concatenate(([False], missing, [False]))
            changes = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1) - 1

            for start, end in zip(starts, ends):
                length = int(end - start + 1)
                duration_ms = length * 1000.0 / sampling_rate_hz
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
                    if duration_ms > longest_partial_ms:
                        longest_partial_ms = duration_ms

                gap_rows.append(
                    {
                        "subject_id": recording["subject_id"],
                        "condition": recording["condition"],
                        "channel": channel,
                        "start_sample": int(start),
                        "end_sample": int(end),
                        "gap_length_samples": length,
                        "gap_duration_ms": duration_ms,
                        "gap_category": category,
                    }
                )

        n_gaps = sum(ordinary.values())
        n_values = n_samples * n_channels
        summary_rows.append(
            {
                "subject_id": recording["subject_id"],
                "condition": recording["condition"],
                "n_gaps": n_gaps,
                "n_single": ordinary["Single"],
                "n_short": ordinary["Short"],
                "n_medium": ordinary["Medium"],
                "n_long": ordinary["Long"],
                "n_complete_channels": n_complete,
                "n_missing_samples": n_missing,
                "missing_percent": (100.0 * n_missing / n_values) if n_values else np.nan,
                "longest_partial_gap_ms": longest_partial_ms,
            }
        )

    gap_summary = pd.DataFrame(summary_rows)
    gap_details = pd.DataFrame(gap_rows)

    labels = [
        f"{row.subject_id} {row.condition}" for row in gap_summary.itertuples(index=False)
    ]
    categories = ["Single", "Short", "Medium", "Long"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(gap_summary))
    for category, color in zip(categories, colors):
        counts = gap_summary[f"n_{category.lower()}"].to_numpy()
        ax.bar(labels, counts, bottom=bottom, color=color, label=category, width=0.7)
        bottom = bottom + counts
    ax.set_xlabel("Recording")
    ax.set_ylabel("Number of gaps")
    ax.set_title("Missing EEG gaps by category")
    ax.legend(title="Gap category")
    fig.tight_layout()

    return gap_summary, gap_details, fig


# endregion

# region STEP 3 - RECONSTRUCT ELIGIBLE MISSING SAMPLES

# WHAT: Estimate missing samples in one recording using linear or PCHIP interpolation.
# WHY: Some short missing segments can be filled for exploratory analysis.
# INPUT: EEG DataFrame plus that recording's gap rows from Step 2.
# OUTPUT: (filled EEG copy, decision log, summary counts).
# RULES: One sample -> linear; 2-250 -> PCHIP; >250 or entire channel -> NaN.
# CAUTION: Medium-gap PCHIP is provisional; all imputation needs validation.
def impute_recording(eeg, gaps):
    filled = eeg.copy()
    n_missing_original = int(eeg.isna().sum().sum())
    log_columns = [
        "subject_id",
        "condition",
        "channel",
        "start_sample",
        "end_sample",
        "gap_category",
        "action",
        "n_samples_filled",
        "reason",
    ]
    if gaps is None or len(gaps) == 0:
        summary = {
            "subject_id": "",
            "condition": "",
            "n_missing_original": n_missing_original,
            "n_linear_samples": 0,
            "n_pchip_samples": 0,
            "n_missing_remaining": n_missing_original,
            "n_skipped_gaps": 0,
            "n_provisional_gaps": 0,
        }
        return filled, pd.DataFrame(columns=log_columns), summary

    subject_id = gaps["subject_id"].iloc[0]
    condition = gaps["condition"].iloc[0]
    log_rows = []
    n_linear = 0
    n_pchip = 0
    n_skipped = 0
    n_provisional = 0

    for channel in eeg.columns:
        channel_gaps = gaps[gaps["channel"] == channel]
        if channel_gaps.empty:
            continue

        observed = eeg[channel].to_numpy(dtype=float)
        values = observed.copy()
        valid_index = np.flatnonzero(~np.isnan(observed))
        pchip = None
# Build PCHIP using only observed samples; never use previously filled values.
        if len(valid_index) >= 2:
            pchip = PchipInterpolator(valid_index, observed[valid_index], extrapolate=False)

        for gap in channel_gaps.itertuples(index=False):
            start = int(gap.start_sample)
            end = int(gap.end_sample)
            category = gap.gap_category
            positions = np.arange(start, end + 1)
            positions = positions[np.isnan(observed[positions])]
            action = "skipped"
            reason = ""
            n_filled = 0
            has_both_sides = (
                len(valid_index) >= 2
                and valid_index[0] < start
                and valid_index[-1] > end
            )

            if category == "Long":
                reason = "long gap left missing"
            elif category == "Complete":
                reason = "complete channel left missing"
            elif len(positions) == 0:
                reason = "gap was not missing in the original signal"
            elif not has_both_sides:
                reason = "no observed sample on both sides"
# For one missing sample, estimate a value on the line between neighbors.
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
# PCHIP uses the shape of nearby observed values; medium gaps are provisional.
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

            log_rows.append(
                {
                    "subject_id": subject_id,
                    "condition": condition,
                    "channel": channel,
                    "start_sample": start,
                    "end_sample": end,
                    "gap_category": category,
                    "action": action,
                    "n_samples_filled": n_filled,
                    "reason": reason,
                }
            )

        filled[channel] = values

    summary = {
        "subject_id": subject_id,
        "condition": condition,
        "n_missing_original": n_missing_original,
        "n_linear_samples": n_linear,
        "n_pchip_samples": n_pchip,
        "n_missing_remaining": int(filled.isna().sum().sum()),
        "n_skipped_gaps": n_skipped,
        "n_provisional_gaps": n_provisional,
    }
    return filled, pd.DataFrame(log_rows), summary



# WHAT: Plot one short and one medium filled gap against the original signal.
# WHY: Visually inspect where interpolation added values and how they join.
# INPUT: Original EEG, imputed EEG, and imputation decision log.
# OUTPUT: One Matplotlib figure with two example panels.
# NOTE: A smooth-looking curve does not prove reconstruction accuracy.
def plot_imputation_example(original, imputed, gap_log):
    short_rows = gap_log[gap_log["action"] == "pchip"]
    medium_rows = gap_log[gap_log["action"] == "provisional_pchip"]
    if short_rows.empty or medium_rows.empty:
        raise ValueError("The gap log needs one filled short gap and one provisional medium gap.")

    shared_channels = sorted(
        set(short_rows["channel"]) & set(medium_rows["channel"]),
        key=lambda name: int(str(name)[2:]),
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
        gap_samples = np.arange(start, end + 1)
        axis.plot(
            samples,
            original[channel].iloc[left:right + 1],
            color="black",
            linewidth=1,
            label="Original",
        )
        axis.plot(
            gap_samples,
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


# endregion

# region STEP 4 - FILTER EEG SIGNALS (1-40 HZ)

# WHAT: Apply a Butterworth 1-40 Hz band-pass filter to finite EEG segments.
# WHY: Reduce slow baseline drift and higher-frequency signal components.
# INPUT: A recording (usually from Step 3) and sampling frequency.
# OUTPUT: (new filtered EEG DataFrame, filtering/too-short segment counts).
# NOTE: A filter attenuates outside-band frequencies; it does not delete samples.
#       NaN gaps remain NaN, and filtering never crosses an unfilled gap.
def bandpass_recording(eeg, sampling_rate_hz=SAMPLING_RATE_HZ):
# Order-4 Butterworth design, stored as stable second-order sections.
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
# Find continuous valid runs: never apply filtfilt across an unfilled NaN gap.
        finite = np.isfinite(values)
        padded = np.concatenate(([False], finite, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        for start, end in zip(starts, ends):
            segment = values[start:end + 1]
            if len(segment) <= pad_length:
                n_too_short += 1
                continue
# Forward/backward filtering removes phase delay within this valid run.
            output[start:end + 1] = sosfiltfilt(sos, segment)
            n_filtered += 1
        filtered[channel] = output

    counts = {
        "n_segments_filtered": n_filtered,
        "n_segments_too_short": n_too_short,
    }
    return filtered, counts

# endregion


# region STEP 5 - SIGNAL ANALYSIS
# These 10-second settings are project choices. Change them here to rerun the metrics.
EPOCH_SECONDS = 10
MAX_IMPUTED_FRACTION = 0.05
WELCH_NPERSEG = 1000
# A boundary belongs to the higher band. Gamma keeps 40 Hz, which is the filter edge.
FREQUENCY_BANDS = (
    ("Delta", 1, 4),
    ("Theta", 4, 8),
    ("Alpha", 8, 13),
    ("Beta", 13, 30),
    ("Gamma", 30, 40),
)
FANO_Z_THRESHOLD = 2.0
FANO_BIN_SECONDS = 0.008


# PURPOSE: Decide which 10-second pieces of a recording can enter the metrics.
# METHOD: Drop channels that are missing for the whole file. Cut non-overlapping
#   10-second epochs only inside stretches where every remaining channel is finite.
#   Stretches are not joined across a NaN gap. A channel may contain at most 5%
#   imputed samples in that epoch. Imputed means NaN in the original file and
#   finite after Step 3.
# INPUT: The original recording, the Step 3 copy, the Step 4 copy, and labels.
# OUTPUT: A report row and a list of accepted epochs. Each epoch is a table of
#   filtered samples for the kept channels.
# PARAMETERS: epoch_seconds=10 and max_imputed_fraction=0.05.
# LIMITATIONS: A stretch shorter than 10 seconds cannot make an epoch, including
#   the short unfiltered runs in 015 Cut. READY means every full block passed.
#   PARTIAL means at least one block was dropped. NOT READY means none passed.
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
    joint = np.isfinite(filtered[kept].to_numpy(dtype=float)).all(axis=1)
    edges = np.diff(np.concatenate(([False], joint, [False])).astype(np.int8))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1) - 1

    epochs = []
    n_short_runs = 0
    n_failed_epochs = 0
    for start, end in zip(run_starts, run_ends):
        if end - start + 1 < epoch_length:
            n_short_runs += 1
            continue
        n_blocks = (end - start + 1) // epoch_length
        for block in range(n_blocks):
            first = start + block * epoch_length
            last = first + epoch_length
            too_imputed = [
                channel
                for channel in kept
                if imputed_flags[channel][first:last].mean() > max_imputed_fraction
            ]
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


# PURPOSE: Measure how much signal power sits in each EEG frequency band.
# METHOD: Welch's method on every channel and every accepted epoch. Absolute
#   power is the area under that band. Relative power divides by the sum of
#   the five band areas. Results stay at epoch level so the notebook can average them.
# INPUT: Epochs from prepare_recording.
# OUTPUT: band_power rows and the matching power spectra from 1 to 40 Hz.
# PARAMETERS: fs=500, nperseg=1000 (0.5 Hz bins). Overlap is SciPy's default,
#   half of nperseg. Bands are delta 1-4, theta 4-8, alpha 8-13, beta 13-30,
#   gamma 30-40 Hz. A shared boundary goes to the higher band, and 40 Hz stays in gamma.
# LIMITATIONS: This is power, not a consciousness score. The 1 Hz and 40 Hz
#   filter edges still shape the spectrum.
def welch_band_power(epochs, sampling_rate_hz=SAMPLING_RATE_HZ, nperseg=WELCH_NPERSEG):
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


# PURPOSE: Estimate how steeply power falls as frequency rises.
# METHOD: Average the Welch spectrum across epochs within each channel, then
#   fit specparam in fixed aperiodic mode. The exponent is positive: a larger
#   value is a steeper fall in log-log space. The same spectrum is fit on
#   1-40 Hz, 1-20 Hz, and 20-40 Hz.
# INPUT: The spectrum table returned by welch_band_power.
# OUTPUT: One exponent per subject, condition, channel, and frequency range,
#   plus the 1-40 Hz aperiodic curve used for the log-log figure.
# PARAMETERS: specparam aperiodic_mode="fixed". Peak limits are only there so
#   the aperiodic fit is not pulled toward a narrow peak.
# LIMITATIONS: This is not a consciousness measure. The 1 Hz and 40 Hz filter
#   edges can bend the fit. 20 Hz is included in both split ranges.
def spectral_exponent(spectra):
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


# PURPOSE: Measure how regular or varied a channel is inside one epoch.
# METHOD: Mark each sample above that epoch's own mean as 1 and the rest as 0,
#   then compute normalized Lempel-Ziv complexity with antropy. Normalization
#   divides by n / log2(n). Epoch values are returned separately so they can
#   be averaged within a recording.
# INPUT: Epochs from prepare_recording.
# OUTPUT: One complexity value per subject, condition, epoch, and channel.
# PARAMETERS: Binarization at the epoch mean. antropy normalize=True.
# LIMITATIONS: This is not PCI and it is not a consciousness measure. The
#   mean threshold is a project choice.
def lempel_ziv_complexity(epochs):
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


# PURPOSE: Describe how evenly large excursions are spread across an epoch.
# METHOD: Z-score each channel inside the epoch. An event starts when the
#   absolute z-score crosses 2; later samples in the same excursion are not
#   counted again. Event starts are summed across channels in 8 ms bins.
#   The Fano factor is the variance of those bin counts divided by their mean.
# INPUT: Epochs from prepare_recording.
# OUTPUT: One Fano factor per epoch, and the bin counts used to draw the trace.
# PARAMETERS: threshold=2 standard deviations and bin_seconds=0.008 (4 samples
#   at 500 Hz). Variance uses the population formula, divide by the number of bins.
# LIMITATIONS: The absolute-value rule is this project's choice. It follows the
#   idea of counting events in Maschke et al. (2024), and it is not the
#   variance of the raw EEG. The result is NaN when an epoch has no events.
def fano_factor(
    epochs,
    sampling_rate_hz=SAMPLING_RATE_HZ,
    threshold=FANO_Z_THRESHOLD,
    bin_seconds=FANO_BIN_SECONDS,
):
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

# endregion
