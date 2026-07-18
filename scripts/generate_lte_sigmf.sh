#!/usr/bin/env bash

set -euo pipefail

base_url="http://nas.destevez.net/~daniel/LTE"
data_root="${1:-data}"

download_recording() {
    local workspace="$1"
    local recording="$2"
    local output_dir="${data_root}/${workspace}"
    mkdir -p "$output_dir"
    for suffix in sigmf-meta sigmf-data; do
        local filename="${recording}.${suffix}"
        echo "Downloading ${filename}"
        curl \
            --fail \
            --location \
            --retry 3 \
            --show-error \
            --output "${output_dir}/${filename}" \
            "${base_url}/${filename}"
    done
}

download_recording "lte/downlink" "LTE_downlink_806MHz_2022-04-09_30720ksps"
download_recording "lte/uplink" "LTE_uplink_847MHz_2022-01-30_30720ksps"

echo "LTE SigMF recordings are available under ${data_root}"
