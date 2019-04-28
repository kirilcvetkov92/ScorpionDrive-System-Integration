## ROS

### running ROS master

```bash
roscore
```

### run ROS node

```bash
rosrun <package> <node_name>
rosrun turtlesim turtlesim_node
```

### list active nodes

```bash
rosnode list
```

### list topic list

```bash
rostopic list
```

### topic info

- pub/sub information
- message type information

```bash
rostopic info <topic_name>

# example
rostopic info /turtle1/cmd_vel
``` 

### message info

```bash
rosmsg info <message_name>

# example
rosmsg info geometry_msgs/Twist
```

### edit message

```bash
rosed <message_cat> <message_name>

# example
rosed geometry_msgs Twist.msg
```

### echo messages on a topic

```bash
rostopic echo <topic_name>

# example
rostopic echo /turtle1/cmd_vel
```

### create catkin workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
cd ..
catkin_make
```

### launch ros node

```bash
roslaunch [package] <filename>

# example
roslaunch simple_arm robot_spawn.launch
```

### check package dependecies

```bash
rosdep check <package>

# example
rosdep check simple_arm
```

### install missing package dependencies

```bash
rosdep install -i <package>

# example
rosdep install -i simple_arm
```

### creating a catkin package

```bash
catkin_create_pkg <your_package_name> [dependency1 dependency2 …]

# example
catkin_create_pkg first_package
```

### run script

```bash
rosrun <package> <script_name>

# example
rosrun simple_arm hello
```

### calling a service
 
```bash
rosservice call /arm_mover/safe_move "joint_1: 1.57
joint_2: 1.57"
```

### set parameter

```bash
rosparam set /arm_mover/max_joint_2_angle 1.57
```

### see logs

```bash
roscd log
```

### filtering /rosout output

```bash
rostopic echo /rosout | grep insert_search_expression_here
```

### running ros bag

```bash
rosbag play -l ../data/dbw_test.rosbag.bag
```

### running tests

```bash
roslaunch ros/src/twist_controller/launch/dbw_test.launch
```

### run all nodes

```bash
roslaunch launch/styx.launch
```

### upgrade test bag

```bash
[ERROR] [1555022619.664346554]: Client [/dbw_test] wants topic /actual/brake_cmd to have datatype/md5sum [dbw_mkz_msgs/BrakeCmd/899b0f3ef31bf0a48497d65b424a1975], but our version has [dbw_mkz_msgs/BrakeCmd/c0d20e1056976680942e85ab0959826c]. Dropping connection.
```

```bash
rosdep update
rosbag fix --force ../data/dbw_test.rosbag.bag ../data/dbw_test.rosbag.new.bag
```