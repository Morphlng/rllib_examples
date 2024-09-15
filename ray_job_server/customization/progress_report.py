from __future__ import annotations

from datetime import datetime

try:
    from ray.tune.experimental.output import AIR_TABULATE_TABLEFMT as TABLEFMT
except ImportError:
    from tabulate import DataRow, Line, TableFormat, tabulate

    TABLEFMT = TableFormat(
        lineabove=Line("╭", "─", "─", "╮"),
        linebelowheader=Line("├", "─", "─", "┤"),
        linebetweenrows=None,
        linebelow=Line("╰", "─", "─", "╯"),
        headerrow=DataRow("│", " ", "│"),
        datarow=DataRow("│", " ", "│"),
        padding=1,
        with_header_hide=None,
    )


def report_progress(
    trial_name: str,
    trial_status: str,
    start_time: str,
    trial_data: dict,
    interval: float = -1,
):
    """Imitate the progress report of a ray.tune trial.

    Args:
        trial_name (str): The name of the trial.
        trial_status (str): The status of the trial.
        start_time (str): The start time of the trial.
        trial_data (dict): The data of the trial.
        interval (int, optional): The time interval between each report. Negative value means no interval. Defaults to -1.
    """

    # Time
    current_time = datetime.now()
    start_time_dt = datetime.fromisoformat(start_time)
    total_running_time = current_time - start_time_dt
    total_seconds = int(total_running_time.total_seconds())
    if interval > 0 and total_seconds < interval:
        return

    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        total_time_str = f"{hours}h {minutes}min {seconds}s"
    elif minutes > 0:
        total_time_str = f"{minutes}min {seconds}s"
    else:
        total_time_str = f"{seconds}s"

    # Print Trial status
    print(f"Trial status: 1 {trial_status}")
    print(f"Current time: {current_time_str}. Total running time: {total_time_str}")

    # Print Logical resource usage
    cpus_used = trial_data.get("cpus_used", 0)
    cpus_total = trial_data.get("cpus_total", 0)
    gpus_used = trial_data.get("gpus_used", 0)
    gpus_total = trial_data.get("gpus_total", 0)
    accelerator_used = trial_data.get("accelerator_used", 0)
    accelerator_total = trial_data.get("accelerator_total", 0)
    accelerator_type = trial_data.get("accelerator_type", "")

    print(
        f"Logical resource usage: {cpus_used}/{cpus_total} CPUs, {gpus_used}/{gpus_total} GPUs "
        f"({accelerator_used}/{accelerator_total} accelerator_type:{accelerator_type})"
    )

    # Exclude resource usage keys from the table as they are already printed
    exclude_keys = {
        "cpus_used",
        "cpus_total",
        "gpus_used",
        "gpus_total",
        "accelerator_used",
        "accelerator_total",
        "accelerator_type",
    }

    # Prepare table headers and rows dynamically
    data_keys = [key for key in trial_data.keys() if key not in exclude_keys]
    headers = ["Trial name", "status"] + data_keys

    # Prepare row values
    row_values = [trial_name, trial_status]

    for key in data_keys:
        value = trial_data[key]
        if isinstance(value, float):
            if key == "total_time_s":
                formatted_value = f"{value:.3f}"
            elif "mean" in key or "avg" in key:
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value}"
        else:
            formatted_value = value
        row_values.append(formatted_value)

    # Create table
    table = tabulate(
        [row_values],
        headers=headers,
        tablefmt=TABLEFMT,
        stralign="center",
    )
    print(table)
