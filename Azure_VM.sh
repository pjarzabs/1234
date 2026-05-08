#!/usr/bin/zsh

az login --use-device-code

az account set -s TWOJ_SUBSCRIPTION_ID

az group create -n LabSK -l polandcentral

az vm create \
    -n piotruspan123 \
    -g LabSK \
    --image Ubuntu2404 \
    --size Standard_B1s \
    --admin-username piotroot \
    --ssh-key-values ~/.ssh/id_ed25519.pub


az account list-locations -o table

az vm list-sizes -l polandcentral -o table
