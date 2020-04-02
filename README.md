# A Behavioral Approach to Visual Navigation with Graph Localization Networks

![GraphNav Overview](https://graphnav.stanford.edu/images/graphnav-architecture-overview.jpeg)

Paper: [A Behavioral Approach to Visual Navigation with Graph Localization Networks](http://arxiv.org/abs/1903.00445)  
Website: https://graphnav.stanford.edu/  
Video: https://youtu.be/nN3B1F90CFM

**Citing**  
```
@INPROCEEDINGS{Savarese-RSS-19, 
    AUTHOR    = {Kevin Chen AND Juan Pablo de Vicente AND Gabriel Sepulveda AND Fei Xia AND Alvaro Soto AND Marynel VÃ¡zquez AND Silvio Savarese}, 
    TITLE     = {A Behavioral Approach to Visual Navigation with Graph Localization Networks}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2019}, 
    ADDRESS   = {FreiburgimBreisgau, Germany}, 
    MONTH     = {June}, 
    DOI       = {10.15607/RSS.2019.XV.010} 
} 
```

## Environment Setup

### GraphNav Setup

```bash
# Need python 2 when using ROS
conda create -n graphnav python=3.7

# Install requirements
pip install -r requirements.txt

# Add graphnav
export PYTHONPATH=$PYTHONPATH:<path-to-graphnav>/src
```

### Gibson Setup

Install Gibson v2. Follow the directions [here](https://github.com/StanfordVL/GibsonEnvV2/tree/master/examples/ros/gibson2-ros) for ROS-specific instructions.

Example Gibson installation below using Anaconda. For more details, see [Gibson](https://github.com/StanfordVL/GibsonEnvV2).
```bash
conda create -n graphnav-gibson-py2 python=2.7
conda activate graphnav-gibson-py2
pip install -e .  # From GibsonEnvV2 directory
conda deactivate
```

If not done already, execute the [ROS-specific Gibson instructions](https://github.com/StanfordVL/GibsonEnvV2/tree/master/examples/ros/gibson2-ros).

Example of path setup below. Put these in a script if you would like.
```bash
export PYTHONPATH=$PYTHONPATH:<path-to-GibsonEnvV2>
export PYTHONPATH=$PYTHONPATH:<path-to-graphnav-gibson-py2-venv>/lib/python2.7/site-packages  # Path to virtual env if you used one (use Python 2)
export PYTHONPATH=$PYTHONPATH:<path-to-graphnav>/src

# Make sure to remove these from PATH as specified in the Gibson ROS instructions (if using Anaconda)
echo $PATH | grep -oP "[^:;]+" | grep conda
```

Add ROS packages for Gibson and graphnav, just like the [ROS-specific Gibson instructions](https://github.com/StanfordVL/GibsonEnvV2/tree/master/examples/ros/gibson2-ros).
```bash
# Gibson
ln -s <path-to-GibsonEnvV2>/examples/ros/gibson-ros/ ~/catkin_ws/src/
cd ~/catkin_ws && catkin_make

ln -s <path-to-graphnav> ~/catkin_ws/src/semnav_ros
cd ~/catkin_ws && catkin_make
```

## Dataset Setup

Download the data (collected from Gibson 1) from [here](http://download.cs.stanford.edu/downloads/graphnav/v0.2.zip). This data is used for training the behaviors networks.

Replace the following in `graphnav/config.py`:
```bash
DATASET_ROOT = '/data/graphnav/trajectory-data'                                                                                                                                                                                            
LOG_ROOT = '/data/graphnav/experiments'                                                                                                                                                                                                    
STANFORD_JSON = '<path-to-graphnav>/data/semantic_labels.json'                                                                                                                                                                     
MAPS_ROOT = '<path-to-graphnav>/maps/v0.2'
 ```

For faster data loader initialization in each run:
```bash
# Edit the directory in build_dataset_cache.py
python build_dataset_cache.py --dataset_type frame_by_frame
```

## Behavior Networks

### Training

Example train command:
```bash
python learning/behavior_net/behavior_trainer.py \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dataset v0.2 \
    --n_workers 3 \
    --behaviornet_type behavior_rnn \
    --dataset_type temporal \
    --print_freq 250 \
    --val_freq 1000 \
    --ckpt_freq 1000 \
    --n_epochs 500 \
    --log_dir v0.2/behavior_rnn/tl \
    --behavior_id tl \
    --n_frames_per_sample 20
```

An example config for different behavior networks:
```
# behaviornet_type, behavior_id, dataset_type
behaviornets = [
        ('behavior_rnn', 'tl', 'temporal'),
        ('behavior_rnn', 'tr', 'temporal'),
        ('behavior_cnn', 'cf', 'temporal'),
        ('behavior_rnn', 's', 'temporal'),
        ('behavior_cnn', 'fd', 'temporal'),
        ]
```

## Graph Localization Network

### Training

```bash
python learning/graph_net/graph_net_trainer.py \
    --dataset v0.2 \
    --learning_rate 1e-4 \
    --n_workers 3 \
    --behaviornet_type graph_net \
    --dataset_type graph_net \
    --print_freq 100 \
    --val_freq 2000 \
    --ckpt_freq 2000 \
    --n_epochs 500 \
    --n_frames_per_sample 20 \
    --log_dir v0.2/graph_net \
    --aggregate_method sum \
    --use_gn_augmentation
```

## Rollouts in Gibson

The code has been tested with ROS Kinetic (Python 2.7).

### Setup for each run

#### Yaml

Edit the Gibson configuration [yaml file](https://github.com/StanfordVL/GibsonEnvV2/blob/master/examples/ros/gibson2-ros/turtlebot_rgbd.yaml). Pay particular attention to the model ID. Also make sure `fov: 2.62` or `150` degrees (depending on Gibson 1 or Gibson 2) and `resolution: 320`.

#### Launch file

Edit the ROS [launch file](https://github.com/StanfordVL/GibsonEnvV2/blob/master/examples/ros/gibson2-ros/launch/turtlebot_gt_navigation.launch). Make sure the [area yaml](https://github.com/StanfordVL/GibsonEnvV2/blob/9dfc340d85e167983df7abfd7425d4ba78ab8524/examples/ros/gibson2-ros/launch/turtlebot_gt_navigation.launch#L42) matches with the model ID specified in the `turtlebot_rgbd.yaml` and points to a yaml from the **this repo**.

#### Load checkpointed models

Edit `graph_net_navigation_brain.py` to load the desired checkpointed models.

### Launching Gibson + ROS

```bash
roslaunch gibson2-ros turtlebot_gt_navigation.launch
rosrun semnav_ros navigation_planner.py
rosrun semnav_ros graph_net_navigation_brain.py --dataset_type graph_net  # No particle filter
```
