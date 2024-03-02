#!/bin/bash

# Default values
CONDA_ENV="ray"
RAY_PORT="6380"
PROMETHEUS_PATH="${PROMETHEUS_PATH:-$HOME/software/prometheus/prometheus}"

# Function to display usage help
usage() {
    echo "Usage: $0 [-e conda_env] [-p ray_port]"
    echo "  -e    Set the conda environment name (default: ray)"
    echo "  -p    Set the Ray head node port number (default: 6380)"
}

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to run commands in a new screen session
run_in_new_screen() {
    if screen -list | grep -q "\.$1\s"; then
        echo "A screen session named '$1' already exists."
    else
        screen -dmS "$1" bash -c "$2"
    fi
}

# Parse command-line options
while getopts ":e:p:h" opt; do
    case ${opt} in
    e)
        CONDA_ENV=$OPTARG
        ;;
    p)
        RAY_PORT=$OPTARG
        ;;
    h)
        usage
        exit 0
        ;;
    \?)
        echo "Invalid option: $OPTARG" 1>&2
        usage
        exit 1
        ;;
    :)
        echo "Invalid option: $OPTARG requires an argument" 1>&2
        usage
        exit 1
        ;;
    esac
done

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Start and stop processes
log "Stopping current Ray instance..."
ray stop
sleep 1

log "Starting Ray head node"
ray start --head --port=$RAY_PORT --include-dashboard=true --dashboard-host=0.0.0.0
if [ $? -ne 0 ]; then
    log "ERROR: Failed to start Ray head node." >&2
    exit 1
fi

log "Starting Prometheus server..."
run_in_new_screen "Prometheus Server" "$PROMETHEUS_PATH --config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"

log "Starting Grafana server..."
log "Note: Grafana needs sudo access to start, please manually enter your password in 'screen -r Grafana Server'"
run_in_new_screen "Grafana Server" "sudo grafana-server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini --homepath /usr/share/grafana web"

log "Script completed successfully."
