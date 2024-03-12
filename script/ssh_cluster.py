from __future__ import annotations

import argparse
import json
import os

import yaml
from fabric import Config, Connection, GroupResult, Result
from fabric import SerialGroup as SGroup
from fabric import ThreadingGroup as TGroup


class SSHCluster:
    def __init__(self, config: "dict | str"):
        """Create a ssh cluster from a configuration file or a config dict."""
        if isinstance(config, dict):
            self.config = config
        elif os.path.isfile(config):
            _, ext = os.path.splitext(config)
            if ext == ".json":
                with open(config, "r") as f:
                    self.config = json.load(f)
            elif ext in [".yml", ".yaml"]:
                with open(config, "r") as f:
                    self.config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration input: {ext}")

        self.node_configs: dict[str, dict] = self.config["nodes"]

        self.nodes: dict[str, Connection] = {}
        self._init_nodes()
        if self.config.get("parallel", False):
            self.group = SGroup.from_connections(list(self.nodes.values()))
        else:
            self.group = TGroup.from_connections(list(self.nodes.values()))

    def __del__(self):
        self.group.close()

    def __len__(self):
        return len(self.group)

    def __iter__(self):
        return self.group.__iter__()

    def __next__(self):
        return self.group.__next__()

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.nodes.keys()
        elif isinstance(key, Connection):
            return key in self.group

        return False

    def _init_nodes(self):
        for name, node_config in self.node_configs.items():
            # Allow stuffing connection details in "host": [user@]host[:port]
            # Or separate them into "user", "host", "port"
            if "host" not in node_config:
                raise ValueError(f"Missing `host` in node {name}")

            host = node_config["host"].strip()
            user = None
            port = 22
            if "@" in host:
                user, host = host.split("@")
                if ":" in host:
                    host, port = host.split(":")
            else:
                user = node_config.get("user", None)
                port = node_config.get("port", 22)

            if user is None:
                raise ValueError(f"Missing `user` in node {name}")

            connect_config = {}
            password = node_config.get("password", None)
            key_filename = node_config.get("key_filename", None)
            pkey = node_config.get("pkey", None)
            if password is not None:
                connect_config["password"] = password
            elif key_filename is not None:
                connect_config["key_filename"] = key_filename
            elif pkey is not None:
                connect_config["pkey"] = pkey

            config = None
            if node_config.get("allow_sudo", False):
                if password is None:
                    raise ValueError(f"Missing `password` for sudo in node {name}")
                config = Config(overrides={"sudo": {"password": password}})

            self.nodes[name] = Connection(
                host=host,
                user=user,
                port=int(port),
                connect_kwargs=connect_config,
                config=config,
            )

    def toggle_concurrency(self, parallel: bool = True):
        """Toggle the concurrency mode of the cluster."""
        if parallel:
            self.group = SGroup.from_connections(list(self.nodes.values()))
        else:
            self.group = TGroup.from_connections(list(self.nodes.values()))

    def run(self, node_name: str, command: str) -> Result:
        """Run a command on a node."""
        if "sudo" in command:
            return self.nodes[node_name].sudo(command)
        else:
            return self.nodes[node_name].run(command)

    def run_all(self, command: str, exclude_nodes: list[str] = None) -> GroupResult:
        """Run a command on all nodes."""
        if exclude_nodes:
            nodes = [
                node for name, node in self.nodes.items() if name not in exclude_nodes
            ]
            group = type(self.group).from_connections(nodes)
        else:
            group = self.group

        if "sudo" in command:
            return group.sudo(command)
        else:
            return group.run(command)

    def upload_file(self, node_name: str, local_path: str, remote_path: str) -> Result:
        """Upload a file to a node."""
        return self.nodes[node_name].put(local_path, remote_path)

    def upload_file_all(
        self, local_path: str, remote_path: str, exclude_nodes: list[str] = None
    ) -> GroupResult:
        """Upload a file to all nodes."""
        if exclude_nodes:
            nodes = [
                node for name, node in self.nodes.items() if name not in exclude_nodes
            ]
            group = type(self.group).from_connections(nodes)
        else:
            group = self.group

        return group.put(local_path, remote_path)

    def download_file(
        self, node_name: str, remote_path: str, local_path: str
    ) -> Result:
        """Download a file from a node."""
        return self.nodes[node_name].get(remote_path, local_path)

    def execute(self, node_name: str, file: str) -> list[Result]:
        """Execute a local script on a node."""
        with open(file, "r") as f:
            script = f.read()

        res = []
        commands = self._parse_commands(script)
        for command in commands:
            res.append(self.run(node_name, command))
        return res

    def execute_all(
        self, file: str, exclude_nodes: list[str] = None
    ) -> list[GroupResult]:
        """Execute a local script on all nodes."""
        with open(file, "r") as f:
            script = f.read()

        res = []
        commands = self._parse_commands(script)
        for command in commands:
            res.append(self.run_all(command, exclude_nodes))
        return res

    def _parse_commands(self, script: str) -> list[str]:
        """Parse the commands from a script content."""
        lines = script.splitlines()

        commands = []
        current_command = ""
        for line in lines:
            line = line.strip()
            if line.endswith("\\"):
                current_command += line[:-1]
                if current_command[-1] != " ":
                    current_command += " "
            else:
                if current_command:
                    commands.append(current_command + line)
                    current_command = ""
                elif line:
                    commands.append(line)
        return commands

    def __repr__(self):
        node_info = {
            node_name: f"{node.user}@{node.host}:{node.port}"
            for node_name, node in self.nodes.items()
        }
        return f"SSHCluster({node_info})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSH Cluster")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="The configuration file to create the cluster.",
        required=True,
    )
    parser.add_argument(
        "-j", "--job", default=None, type=str, help="The jobs to be executed."
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode."
    )
    args = parser.parse_args()

    cluster = SSHCluster(args.config)

    if args.interactive or args.job is None:
        user_input = ""
        share_command = False
        exit_commands = ["exit", "quit"]
        cluster_name = ["all", "cluster", "share"]

        while True:
            user_input = input(">>> Target: ")
            if user_input in exit_commands:
                break

            if user_input in cluster:
                target = user_input
                share_command = False
            elif user_input in cluster_name:
                share_command = True
            elif user_input == "":
                continue
            else:
                print(f"Invalid target: {user_input}")
                print(f"Available targets: {list(cluster.nodes.keys()) + cluster_name}")
                continue

            while True:
                func = (
                    cluster.run_all
                    if share_command
                    else lambda x: cluster.run(target, x)
                )

                user_input = input(">>> Command: ")
                if user_input in exit_commands:
                    break

                if "upload" in user_input:
                    # Try to parse local and remote paths
                    if user_input == "upload":
                        local = input("Local path: ")
                        remote = input("Remote path: ")
                    else:
                        try:
                            _, local, remote = user_input.split()
                        except ValueError:
                            print("Invalid upload command.")
                            continue
                    user_input = (local, remote)
                    if share_command:
                        func = lambda x: cluster.upload_file_all(*x)
                    else:
                        func = lambda x: cluster.upload_file(target, *x)
                elif "download" in user_input:
                    if share_command:
                        print("Download is not supported in share mode.")
                        continue
                    if user_input == "download":
                        remote = input("Remote path: ")
                        local = input("Local path: ")
                    else:
                        try:
                            _, remote, local = user_input.split()
                        except ValueError:
                            print("Invalid download command.")
                            continue
                    user_input = (remote, local)
                    func = lambda x: cluster.download_file(target, *x)

                try:
                    res = func(user_input)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
    print("Bye!")
