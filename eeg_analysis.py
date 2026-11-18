"""Load and inspect the EEG recordings.

Each CSV in the data folder is one full recording. The file name gives the
subject and the condition. Files named "surgery" are the first incision,
stored here as Cut.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

SAMPLING_RATE_HZ = 500
CONDITION_ORDER = {"Baseline": 0, "IFT": 1, "Cut": 2}

# Gap-length cutoffs for this project, not values taken from the papers.
# At 500 Hz, 50 samples is 100 ms and 250 samples is 500 ms.
GAP_SHORT_MAX = 50
GAP_MEDIUM_MAX = 250


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


# Finds each run of consecutive missing samples and labels the run by length.
# Gap length decides which imputation method is reasonable later, and a complete channel is not the same as a short gap.
# Returns the summary table, the detailed gap table, and one stacked bar chart.
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


# Fills single, short, and medium gaps in one recording and leaves the rest missing.
# Long gaps and empty channels cannot be recovered from neighboring samples in that channel.
# Returns a copy of the recording, a log of every gap, and one summary row.
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


# Draws the original signal and the filled samples around one short gap and one medium gap.
# One example is enough to see what changed, without a separate plot for every channel.
# Returns a figure with two panels. Medium-gap values are marked as provisional.
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
