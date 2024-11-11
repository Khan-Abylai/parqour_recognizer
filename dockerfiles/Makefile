mnt=/home/imangali
mnt_ser=/mnt/d/Ubuntu/General/recognizer-training-pipeline/data
url=0.0.0.0:4045:22
name=trainer_container1
run:
		docker run -d -it --ipc=host --cap-add sys_ptrace -p ${url} \
		-v ${mnt_ser}:/home/user/code \
		-v ${mnt}:/home/user/mnt --gpus '"device=0"' --name ${name} \
		-e NCCL_IB_DISABLE=1 \
		-e NCCL_P2P_DISABLE=1 \
		-e NCCL_SOCKET_IFNAME=eth0 \
		registry.infra.smartparking.kz/trainer:1.0
stop:
		docker stop ${name}
restart:
		make stop
		make run
