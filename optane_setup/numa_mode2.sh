sudo daxctl migrate-device-model
sudo ndctl create-namespace --mode=devdax --map=mem
sudo ndctl create-namespace --mode=devdax --map=mem
sudo daxctl reconfigure-device dax1.0 --mode=system-ram
sudo daxctl reconfigure-device dax0.0 --mode=system-ram
numactl -H
