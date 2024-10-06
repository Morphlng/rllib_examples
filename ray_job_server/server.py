from __future__ import annotations

import base64
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import ray
from fastapi import FastAPI
from ray.job_submission import JobSubmissionClient

import cluster.utils as cu

app = FastAPI()


@dataclass
class ServerConfig:
    ray_address: str = None
    """Address of the Ray cluster head node. If set to 'auto', will use the default address."""

    dashboard_address: str = None
    """Address of the Ray dashboard. If not set, will use the Ray address."""

    tensorboard_address: str = None
    """Host address of the tensorboard server."""

    ray_storage: str = None
    """Path to the Ray storage directory."""

    ray_log_storage: str = None
    """Path to the log storage directory."""

    job_working_dir: str = None
    """Path to the submiting job working directory."""

    log_dir: str = None
    """Path to the JobServer log directory."""

    excludes: list = None
    """List of directories to exclude from the job submission."""

    def __post_init__(self):
        # fmt: off
        if self.ray_address is None:
            self.ray_address = os.environ.get("RAY_ADDRESS", "auto")
        if self.dashboard_address is None:
            self.dashboard_address = os.environ.get("DASHBOARD_ADDRESS", self.ray_address)
        if self.tensorboard_address is None:
            self.tensorboard_address = os.environ.get("TENSORBOARD_ADDRESS", "http://localhost:6006")
        if self.ray_storage is None:
            self.ray_storage = os.environ.get("RAY_STORAGE", "/mnt/ray/ray_results")
        if self.ray_log_storage is None:
            self.ray_log_storage = os.environ.get("RAY_LOG_STORAGE", "/mnt/ray/ray_logs")
        if self.job_working_dir is None:
            self.job_working_dir = os.path.dirname(__file__)
        if self.log_dir is None:
            self.log_dir = "logs/"
        if self.excludes is None:
            self.excludes = [
                "__pycache__/*",
                "data/*",
                "docker/*",
                "logs/*",
                "scripts/*",
            ]
        # fmt: on
        self.runtime_env = {
            "working_dir": self.job_working_dir,
            "excludes": self.excludes,
        }

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class JobServer:
    """Singleton class to manage the Ray job submission server."""

    _initialized = False
    logger = None
    client = None
    config = ServerConfig()

    @staticmethod
    def init(config: Optional[dict] = None):
        if JobServer._initialized:
            return

        if config is not None:
            JobServer.config.update(**config)

        JobServer.logger = JobServer._setup_logger()
        JobServer.logger.info(
            "Initializing Job Server with config: \n%s", JobServer.config
        )
        JobServer.client = JobSubmissionClient(
            address=JobServer.config.dashboard_address
        )
        if not ray.is_initialized():
            ray.init(
                address=JobServer.config.ray_address,
                runtime_env=JobServer.config.runtime_env,
            )

        JobServer._initialized = True

    @staticmethod
    @app.get("/task/submit/{task_config}")
    def submit_job(task_config: str) -> str:
        """Submit a job to the Ray cluster.

        Args:
            task_config (str): Base64 encoded JSON string of the task configuration.

        Returns:
            str: job_id of the submitted job.
        """
        decode_str = base64.b64decode(task_config).decode("utf-8")
        config: dict = json.loads(decode_str)
        config.update({"run_config": {"storage": JobServer.config.ray_storage}})

        task_id = config.get("task_id", None)
        if JobServer.get_job_status(task_id) is not None:
            JobServer.logger.warning(
                f"Task '{task_id}' is already in cluster, will generate a new task_id."
            )
            task_id = None

        submission_id = JobServer.client.submit_job(
            entrypoint=f"python runner.py --config '{json.dumps(config)}'",
            submission_id=task_id,
            runtime_env=JobServer.config.runtime_env,
        )
        return submission_id

    @staticmethod
    @app.get("/task/resume/{task_config}")
    def resume_job(task_config: str):
        """Resume a job in the Ray cluster.

        Args:
            task_config (str): Base64 encoded JSON string of the task configuration.

        Returns:
            str: job_id of the resumed job.
        """
        decode_str = base64.b64decode(task_config).decode("utf-8")
        config: dict = json.loads(decode_str)
        config.update(
            {"resume": True, "run_config": {"storage": JobServer.config.ray_storage}}
        )

        submission_id = JobServer.client.submit_job(
            entrypoint=f"python runner.py --config '{json.dumps(config)}'",
            runtime_env=JobServer.config.runtime_env,
        )
        return submission_id

    @staticmethod
    @app.get("/task/stop/{job_id}")
    def stop_job(job_id: str) -> bool:
        """Stop a job in the Ray cluster.

        Args:
            job_id (str): job_id of the job to stop.

        Returns:
            bool: True if the job is interrupted, False otherwise.
        """
        try:
            state = JobServer.client.stop_job(job_id)
        except Exception as e:
            JobServer.logger.error(e)
            state = False

        return state

    @staticmethod
    @app.get("/task/delete/{job_id}")
    def delete_job(job_id: str) -> bool:
        """Delete a job in the Ray cluster.

        Args:
            job_id (str): job_id of the job to delete.

        Returns:
            bool: True if the job is deleted successfully, False otherwise.

        Warning:
            Delete job does not delete corresponding logs and checkpoints.
        """
        try:
            state = JobServer.client.delete_job(job_id)
        except Exception as e:
            JobServer.logger.error(e)
            state = False

        return state

    @staticmethod
    @app.get("/task/log/{job_id}")
    def get_job_log(job_id: str):
        """Get the logs of a job in the Ray cluster.

        Args:
            job_id (str): job_id of the job to get status.

        Returns:
            str: logs of the job, None if the job is not found.
        """
        try:
            logs = JobServer.client.get_job_logs(job_id)
        except RuntimeError as e:
            JobServer.logger.warning(
                "Job not found in current cluster session."
                f"Trying to fetch from {JobServer.config.ray_log_storage}."
            )
            task = cu.get_job_log.remote(job_id, JobServer.config.ray_log_storage)
            result = ray.get(task)
            if result["status"] == "error":
                JobServer.logger.error(f"Job '{job_id}' not found in log storage.")
                logs = "NOT_FOUND"
            else:
                logs = result["data"]

        return logs

    @staticmethod
    @app.get("/task/progress/{job_id}")
    def get_job_progress(job_id: str):
        """Get the progress of a job.

        Args:
            job_id (str): job_id of the job to get progress.

        Returns:
            dict: progress of the job, None if the job is not found.
        """
        log_data = JobServer.get_job_log(job_id)
        if log_data == "NOT_FOUND":
            return None
        else:
            JobServer.logger.info(f"Slicing log data for job {job_id}")
            log_data = log_data[-20000:]

        latest_trial_info = cu.parse_log(log_data)
        trial_details = latest_trial_info.get("trial_details", None)
        if trial_details and isinstance(trial_details, list):
            trial_details = trial_details[0]
        return trial_details

    @staticmethod
    @app.get("/task/result/{task_id}")
    def get_job_result(task_id: str):
        """Get the result (pandas data) of a job.

        Args:
            task_id (str): task_id of the job to get result.

        Returns:
            str: result of the job in JSON format. If the job is not found, return None.
        """
        task = cu.get_job_result.remote(task_id, JobServer.config.ray_storage)
        result = ray.get(task)
        if result["status"] == "error":
            JobServer.logger.error(result["data"])
            return None
        else:
            result = result["data"]

        return result

    @staticmethod
    @app.get("/task/visualize/{task_id}")
    def get_tensorboard_url(task_id: str):
        """Get the URL of tensorboard for visualizing the job.

        Args:
            task_id (str): task_id of the job to visualize.

        Returns:
            str: URL of tensorboard.
        """
        url_template = "{}/#scalars&regexInput={}"
        return url_template.format(JobServer.config.tensorboard_address, task_id)

    @staticmethod
    @app.get("/agent/export/{checkpoint_dir}")
    def export_model(
        checkpoint_dir: str,
        save_dir: str = "L21udC9yYXkvZXhwb3J0X21vZGVscw==",
        onnx: int = None,
    ):
        """Export the agent model from the checkpoint.

        Args:
            checkpoint_dir (str): Base64 encoded checkpoint directory of the agent model.
            save_dir (str): Base64 encoded directory to save the exported model.
            onnx (int): Export the model to ONNX format if specified version, otherwise PyTorch format.

        Returns:
            bool: True if the model is exported successfully, False otherwise.
        """
        try:
            checkpoint_dir = base64.b64decode(checkpoint_dir).decode("utf-8")
            save_dir = base64.b64decode(save_dir).decode("utf-8")
            task = cu.export_model.remote(checkpoint_dir, save_dir, onnx)
            result = ray.get(task)
            saved = result["status"] == "success"
            if not saved:
                JobServer.logger.error(result["data"])
        except Exception as e:
            JobServer.logger.error(e)
            saved = False

        return saved

    @staticmethod
    @app.get("/task/list")
    def list_jobs():
        """List all jobs in the Ray cluster.

        Returns:
            list[dict]: Information of all jobs in the Ray cluster.
        """
        return JobServer.client.list_jobs()

    @staticmethod
    @app.get("/task/checkpoint/{task_id}")
    def list_checkpoints(task_id: str):
        """List all checkpoints of a job in the Ray cluster.

        Args:
            task_id (str): task_id of the job to get checkpoints.

        Returns:
            list[str]: List of checkpoint files.
        """
        try:
            task = cu.get_job_chekckpoints.remote(task_id, JobServer.config.ray_storage)
            result = ray.get(task)
            if result["status"] == "error":
                JobServer.logger.error(result["data"])
                checkpoints = None
            else:
                checkpoints = result["data"]
        except Exception as e:
            JobServer.logger.error(e)
            checkpoints = None

        return checkpoints

    @staticmethod
    @app.get("/task/status/{job_id}")
    def get_job_status(job_id: str):
        """Get the status of a job in the Ray cluster.

        Args:
            job_id (str): job_id of the job to get status.

        Returns:
            str: status of the job, None if the job is not found.
        """
        try:
            status = JobServer.client.get_job_status(job_id)
            if status == "SUCCEEDED":
                log = JobServer.get_job_log(job_id)[-20000:]
                for word in ["TuneError", "ERROR"]:
                    if word in log:
                        status = "FAILED"
                        break
        except Exception as e:
            JobServer.logger.error(e)
            status = None

        return status

    @staticmethod
    @app.get("/task/info/{job_id}")
    def get_job_info(job_id: str):
        """Get the information of a job in the Ray cluster.

        Args:
            job_id (str): job_id of the job to get information.

        Returns:
            dict: Information of the job.
        """
        try:
            info = JobServer.client.get_job_info(job_id)
        except Exception as e:
            JobServer.logger.error(e)
            info = None

        return info

    @staticmethod
    @app.get("/cluster/node/{node_ip}")
    def get_node_url(node_ip: str):
        """Get the URL of a node in the Ray cluster.

        Args:
            node_ip (str): IP address of the node.

        Returns:
            str: URL of the node info page.
        """
        nodes = ray.nodes()
        url_template = "{}/#/cluster/nodes/{}"

        for node in nodes:
            if node["NodeManagerAddress"] == node_ip:
                return url_template.format(
                    JobServer.client.get_address(), node["NodeID"]
                )

        return None

    @staticmethod
    @app.get("/cluster/info")
    def get_cluster_info():
        """Get the information of the Ray cluster.

        Returns:
            list[dict]: Information of each node in the Ray cluster.
        """
        return ray.nodes()

    @staticmethod
    @app.get("/cluster/resources")
    def get_cluster_resources(all: bool = False):
        """Get the resources of the Ray cluster.

        Args:
            all (bool): Get all resources if True, otherwise get available resources.

        Returns:
            dict: Resources of the Ray cluster
        """
        return ray.cluster_resources() if all else ray.available_resources()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        if not os.path.isdir(JobServer.config.log_dir):
            os.mkdir(JobServer.config.log_dir)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(module)s %(funcName)s: "%(message)s"',
            datefmt="%d-%m-%y %H:%M:%S",
        )
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(JobServer.config.log_dir, "job_server.log"),
            "w+",
            maxBytes=10485760,
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger


class RayConn(threading.Thread):
    def __init__(self, timeout: float = 10):
        super().__init__(daemon=True)
        self.timeout = timeout
        self.reconnect_count = 0
        self.start()

    def run(self):
        while True:
            time.sleep(self.timeout)
            if not self._is_ray_connected():
                self._reconnect()

    def _is_ray_connected(self):
        try:
            ray.nodes()
            return True
        except BaseException as e:
            return False

    def _reconnect(self):
        JobServer.logger.error("Ray client is disconnected. Trying to reconnect")
        try:
            self._shutdown_ray()
            JobServer.client = JobSubmissionClient(
                address=JobServer.config.dashboard_address
            )
            ray.init(
                address=JobServer.config.ray_address,
                runtime_env=JobServer.config.runtime_env,
            )
            self.reconnect_count += 1
            JobServer.logger.info(
                f"Successfully reconnected, reconnect count: {self.reconnect_count}"
            )
        except BaseException as e:
            JobServer.logger.error(f"Failed to connect to Ray head: {e}")

    def _shutdown_ray(self):
        try:
            ray.shutdown()
            JobServer.logger.info("Ray shutdown complete.")
        except BaseException as e:
            JobServer.logger.error(f"Failed to shutdown Ray: {e}")


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Ray Job Submission Server")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    parser.add_argument("--timeout", type=float, default=5, help="Reconnect timeout")
    parser.add_argument("--port", type=int, default=8000, help="JobServer Port number")
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            args.config = json.load(f)

    JobServer.init(args.config)
    RayConn(args.timeout)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
