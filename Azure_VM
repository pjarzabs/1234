#!/usr/bin/zsh
RG=LabSK LOC=swedencentral VM=jarzabski-u24-vm DNS=jarzabski-u24

az login --use-device-code
az account set -s 7376751c-7b85-4428-8ba3-23f179599e5f
az group create -n $RG -l $LOC
az vm create -g $RG -n $VM -l $LOC \
  --image Ubuntu2404 --size Standard_D2s_v3 \
  --admin-username $USER --ssh-key-value ~/.ssh/id_ed25519.pub
az network public-ip update -g $RG -n ${VM}PublicIP --dns-name $DNS

echo ssh $USER@$(az vm show -d -g $RG -n $VM --query publicIps -o tsv)

az vm deallocate -g $RG -n $VM
