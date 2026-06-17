Download docker Image and create Container:
```bash
### pull the docker Image from dockerhub:
docker pull yipko/x-icm:snapshot-20251031

### initialize the container (the port of ssh will be 6666)
sudo docker run -it -v /tmp/.X11-unix/:/tmp/.X11-unix:rw -v /usr/lib/nvidia:/usr/lib/nvidia -e SDL_VIDEO_GL_DRIVER=libGL.so.1.7.0 -e DISPLAY=$DISPLAY -e NVIDIA_VISIBLE_DEVICES=all -e  NVIDIA_DRIVER_CAPABILITIES=all --gpus=all -p 6666:22 -v --name x-icm yipko/x-icm:snapshot-20251031
```

After loading, enter the Pixi environment:
```bash
cd X-ICM && git pull # keep the X-ICM repo up-to-date
pixi shell
```
