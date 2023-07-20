# cuda_camera_undistort
Nvivafilter plugin for cata camera undistort based on jetson platform

### Building from Source

```shell
mkdir build
cd build
cmake -DMAP_X_PATH=/opt/pmtd/config/lidar_camera_config/pmtd104/map_X.yaml -DMAP_Y_PATH=/opt/pmtd/config/lidar_camera_config/pmtd104/map_Y.yaml ..
make 
cp libcuda_camera_undistort.so /opt/pmtd/lib/
```

### Usage

### Command line

```shell
gst-launch-1.0 nvv4l2camerasrc device=/dev/video0 ! "video/x-raw(memory:NVMM),format=(string)UYVY, width=(int)1920, height=(int)1080" ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)I420" ! nvvidconv ! nvivafilter customer-lib-name="/opt/pmtd/lib/libcuda_camera_undistort.so" cuda-process=true ! "video/x-raw(memory:NVMM),format=(string)RGBA" ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)I420" ! nv3dsink
```



