DIR=$(pwd)/../
xhost +local:1000

if docker ps -a --format '{{.Names}}' | grep -q '^bundlesdf$'; then
    echo "Attaching to existing bundlesdf container..."
    docker start bundlesdf 2>/dev/null
    docker exec -it -e DISPLAY="$DISPLAY" -w "$DIR" bundlesdf bash
else
    echo "Starting new bundlesdf container..."
    docker run \
        --name bundlesdf \
        --gpus all \
        --env NVIDIA_DISABLE_REQUIRE=1 \
        -it \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "$DIR:$DIR" \
        -v /home:/home \
        -v /mnt:/mnt \
        -v /tmp:/tmp \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
        --network=host \
        --ipc=host \
        -e DISPLAY="$DISPLAY" \
        -e GIT_INDEX_FILE \
        -w "$DIR" \
        nvcr.io/nvidian/bundlesdf:latest bash -c "cd $DIR && bash"
fi
