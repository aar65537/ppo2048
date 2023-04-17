sudo semanage boolean -m --on container_use_devices
sudo semodule -i .devcontainer/cuda_container.cil /usr/share/udica/templates/base_container.cil