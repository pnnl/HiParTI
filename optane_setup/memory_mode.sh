sudo ndctl disable-namespace namespace0.0
sudo ndctl disable-namespace namespace1.0
sudo ndctl destroy-namespace namespace0.0 --force
sudo ndctl destroy-namespace namespace1.0 --force
sudo ipmctl delete -goal
echo "y" | sudo ipmctl create -goal memorymode=100
sudo reboot
