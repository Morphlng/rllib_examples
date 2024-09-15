from __future__ import annotations

import glob
import os
import re

import ray


def parse_log(log_data: str):
    """Parse the log data of a Ray job."""
    parsed_info = {}

    log_parts = log_data.split("Trial status:")
    if len(log_parts) > 1:
        log_data = "Trial status:" + log_parts[-1]

    # Split log_data into lines
    lines = log_data.split("\n")

    # Trial status
    for line in lines:
        trial_status_match = re.search(r"Trial status: (\d+) (\w+)", line)
        if trial_status_match:
            parsed_info["trial_count"] = int(trial_status_match.group(1))
            parsed_info["trial_status"] = trial_status_match.group(2)

    # Current time & Total running time
    for line in lines:
        time_match = re.search(
            r"Current time: ([\d-]+ [\d:]+)\. Total running time: (.+)", line
        )
        if time_match:
            parsed_info["current_time"] = time_match.group(1)
            parsed_info["total_running_time"] = time_match.group(2)

    # Logical resource usage
    for line in lines:
        resource_match = re.search(
            r"Logical resource usage: ([\d.]+)/([\d.]+) CPUs, ([\d.]+)/([\d.]+) GPUs",
            line,
        )
        if resource_match:
            parsed_info["cpu_usage"] = float(resource_match.group(1))
            parsed_info["cpu_total"] = float(resource_match.group(2))
            parsed_info["gpu_usage"] = float(resource_match.group(3))
            parsed_info["gpu_total"] = float(resource_match.group(4))

    # Find the header line index
    header_line_index = None
    for i, line in enumerate(lines):
        if "Trial name" in line:
            header_line_index = i
            break

    if header_line_index is not None:
        header_line = lines[header_line_index]
        header_line_content = header_line.strip("│| ")
        headers = re.split(r"\s{2,}", header_line_content)

        trial_details = []
        data_line_index = header_line_index + 2  # Skip the separator line
        while data_line_index < len(lines):
            data_line = lines[data_line_index].strip()
            if (
                not data_line
                or data_line.startswith("╰")
                or data_line.startswith("Trial status:")
            ):
                break
            if data_line.startswith("│") or data_line.startswith("|"):
                data_line_content = data_line.strip("│| ")
                data_fields = re.split(r"\s{2,}", data_line_content)
                if len(data_fields) == len(headers):
                    row_dict = dict(zip(headers, data_fields))
                    for key, value in row_dict.items():
                        try:
                            if "." in value or "e" in value.lower() or "-" in value:
                                row_dict[key] = float(value)
                            else:
                                row_dict[key] = int(value)
                        except ValueError:
                            # Keep as string
                            pass
                    row_dict["iter"] = row_dict.get("iter", 0)
                    trial_details.append(row_dict)
            data_line_index += 1

        parsed_info["trial_details"] = trial_details

    return parsed_info


@ray.remote
def get_job_log(job_id: str, log_dir: str = "/tmp/ray/session_*/logs"):
    """Get the log of a job from the log file.

    Args:
        job_id (str): Job ID to get log.
        log_dir (str): Directory of the log files.

    Returns:
        dict: status and logs of the job.
    """

    log_files = glob.glob(os.path.join(log_dir, f"job-driver-{job_id}.log"))
    if log_files:
        with open(log_files[0], "r") as f:
            logs = f.read()
    else:
        logs = f"Log file for job '{job_id}' not found."

    return {"status": "success" if log_files else "error", "data": logs}


@ray.remote
def get_job_result(task_id: str, ray_storage: str = "/home/ray/ray_results") -> str:
    """Get the result (pandas data) of a job.

    Args:
        task_id (str): task_id of the job to get result.

    Returns:
        str: status and result of the job in JSON string.
    """
    from ray.train import Result

    try:
        ray_storage = os.path.expanduser(ray_storage)
        experiment_path = os.path.join(ray_storage, task_id)
        if not os.path.exists(experiment_path):
            raise FileExistsError(f"Experiment '{task_id}' not found in '{ray_storage}'.")

        trials = [
            os.path.join(experiment_path, d)
            for d in os.listdir(experiment_path)
            if os.path.isdir(os.path.join(experiment_path, d))
        ]
        if not trials:
            raise FileNotFoundError(f"No trials found for experiment '{task_id}'.")

        latest_trial = max(trials, key=lambda t: os.path.getmtime(t))
        result = Result.from_path(latest_trial)
        result = result.metrics_dataframe.to_json()
        status = "success"
    except Exception as e:
        status = "error"
        result = str(e)

    return {"status": status, "data": result}


@ray.remote
def get_job_chekckpoints(
    task_id: str,
    ray_storage: str = "/mnt/ray/ray_results",
    latest_only: bool = True,
):
    """Return all checkpoints paths of a job.

    Args:
        task_id (str): task_id of the job to get checkpoints.
        ray_storage (str): Path to the Ray storage directory.
        latest_only (bool): Return only the latest checkpoint path.

    Returns:
        dict: status and checkpoints paths.
    """
    try:
        ray_storage = os.path.expanduser(ray_storage)
        experiment_path = os.path.join(ray_storage, task_id)
        if not os.path.exists(experiment_path):
            raise FileExistsError(f"Experiment '{task_id}' not found.")

        trials = [
            os.path.join(experiment_path, d)
            for d in os.listdir(experiment_path)
            if os.path.isdir(os.path.join(experiment_path, d))
        ]
        if latest_only:
            trials = [max(trials, key=lambda t: os.path.getmtime(t))]

        checkpoints = []
        for trial in trials:
            checkpoints.extend(glob.glob(os.path.join(trial, "checkpoint_*")))
        status = "success"
    except Exception as e:
        status = "error"
        checkpoints = str(e)
    return {"status": status, "data": checkpoints}


@ray.remote
def export_model(
    checkpoint_dir: str,
    save_dir: str = "/mnt/ray/export_models",
    onnx: int = None,
):
    """Export the agent model from the checkpoint.

    Args:
        checkpoint_dir (str): Checkpoint directory of the agent model.
        save_dir (str): Directory to save the exported model.
        onnx (int): Export the model to ONNX format if specified version, otherwise PyTorch format.

    Returns:
        str: Path to the exported model. Error message if failed.
    """
    from ray.rllib import Policy

    try:
        policy = Policy.from_checkpoint(checkpoint_dir)
        if isinstance(policy, dict):
            for name, p in policy.items():
                p.export_model(f"{save_dir}/{name}", onnx=onnx)
        else:
            policy.export_model(save_dir, onnx=onnx)
        result = save_dir
        status = "success"
    except Exception as e:
        status = "error"
        result = str(e)

    return {"status": status, "data": result}
