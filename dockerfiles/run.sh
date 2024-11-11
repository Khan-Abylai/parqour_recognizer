docker run -d -it --rm --ipc=host --cap-add sys_ptrace -p0.0.0.0:3030:22 \
            --gpus all \
            --volume /mnt/sdb1/LP_RECOGNIZER_DATA:/home/user/mnt \
            --name lp_recognizer \
            registry.infra.smartparking.kz/detector:dev
