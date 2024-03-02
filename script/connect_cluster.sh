#!/bin/bash

# Function to log messages with timestamps
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to display help message
usage() {
    echo "Usage: $0 [-a HEAD_NODE_IP] [-p HEAD_NODE_PORT] [-n NFS_SHARE_PATH] [-e CONDA_ENV_NAME] [-m MOUNT_POINT]"
    echo "  -a HEAD_NODE_IP         IP address of the head node (default: 192.168.16.36)"
    echo "  -p HEAD_NODE_PORT       Port of the head node (default: 6380)"
    echo "  -n NFS_SHARE_PATH       Path to the NFS share, if different from the head node IP (default uses head node IP)"
    echo "  -e CONDA_ENV_NAME       Name of the Conda environment to activate (default: ray)"
    echo "  -m MOUNT_POINT          Mount point for NFS share (default: /mnt/nfs_share)"
    exit 1
}

# Default values
default_head_node_ip="192.168.16.36"
default_head_node_port="6380"
default_nfs_share_base="/mnt/nfs_share" # Base path for NFS share
default_conda_env_name="ray"
default_mount_point="/mnt/nfs_share"

# Initialize variables to default values
head_node_ip=$default_head_node_ip
head_node_port=$default_head_node_port
nfs_share_path="" # Will be set after command line arguments are parsed
conda_env_name=$default_conda_env_name
mount_point=$default_mount_point

# Read command line arguments
while getopts ":a:p:n:e:m:h" opt; do
    case ${opt} in
    a)
        head_node_ip=$OPTARG
        ;;
    p)
        head_node_port=$OPTARG
        ;;
    n)
        nfs_share_path=$OPTARG
        ;;
    e)
        conda_env_name=$OPTARG
        ;;
    m)
        mount_point=$OPTARG
        ;;
    h)
        usage
        ;;
    \?)
        echo "Invalid option: $OPTARG" 1>&2
        usage
        ;;
    :)
        echo "Invalid option: $OPTARG requires an argument" 1>&2
        usage
        ;;
    esac
done
shift $((OPTIND - 1))

# Apply default values if not set
head_node_address="$head_node_ip:$head_node_port" # Combine IP and port
nfs_share_path=${nfs_share_path:-"$head_node_ip:$default_nfs_share_base"}

# Environment setup
eval "$(conda shell.bash hook)"
conda activate "$conda_env_name"

log "Stopping current Ray instance..."
ray stop
sleep 1

log "Connecting to head node at $head_node_address..."
ray start --address="$head_node_address"
if [ $? -ne 0 ]; then
    log "ERROR: Failed to connect to head node." >&2
    exit 1
fi

log "Checking if NFS share is already mounted..."
if mount | grep -q "$mount_point"; then
    log "INFO: NFS share is already mounted."
else
    log "Mounting NFS share..."
    sudo mount "$nfs_share_path" "$mount_point"
    if [ $? -ne 0 ]; then
        log "ERROR: Failed to mount NFS share." >&2
        exit 1
    fi
fi

log "Script completed successfully."
