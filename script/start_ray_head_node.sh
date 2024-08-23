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

# Function to get IP addresses
get_ip_addresses() {
    ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'
}

# Function to select IP address
select_ip_address() {
    local ips=($(get_ip_addresses))
    local default_ip=$(echo "${ips[@]}" | tr ' ' '\n' | grep '^192\.168' | head -n 1)

    if [ ${#ips[@]} -eq 0 ]; then
        echo "No IP addresses found." >&2
        exit 1
    elif [ ${#ips[@]} -eq 1 ]; then
        echo "${ips[0]}"
    else
        if [ -n "$default_ip" ]; then
            echo "Multiple IP addresses found. Using default IP: $default_ip" >&2
            echo "$default_ip"
        else
            echo "Multiple IP addresses found. Please choose one:" >&2
            select ip in "${ips[@]}"; do
                if [ -n "$ip" ]; then
                    echo "$ip"
                    break
                fi
            done
        fi
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

# Get the IP address
MAIN_IP=$(select_ip_address)
if [ -z "$MAIN_IP" ]; then
    log "Failed to select an IP address. Exiting."
    exit 1
fi

# Set environment variables for Grafana and Prometheus
export RAY_GRAFANA_HOST="http://${MAIN_IP}:3000"
export RAY_PROMETHEUS_HOST="http://${MAIN_IP}:9090"

log "Using IP address: $MAIN_IP"
log "Grafana will be available at: $RAY_GRAFANA_HOST"
log "Prometheus will be available at: $RAY_PROMETHEUS_HOST"

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
