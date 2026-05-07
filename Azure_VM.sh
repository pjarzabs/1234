#!/usr/bin/zsh

az login --use-device-code
az account set -s 7376751c-7b85-4428-8ba3-23f179599e5f

az group create -n LabSK -l swedencentral

az vm create \
  -g LabSK \
  -n jarzabski-u24-vm \
  --image Ubuntu2404 \
  --admin-username $USER \
  --ssh-key-value ~/.ssh/id_ed25519.pub
