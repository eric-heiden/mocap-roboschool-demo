# Motion Capture Data in Roboschool
Load ASF/AMC motion capture data into Roboschool/MuJoCo. Subjects (skeletons) provided as ASF files are converted to the MuJoCo XML model format MJCF. Given this model definition, joint animations can be read from AMC files to be played back in the Roboschool gym (deactivated physics simulation). An example from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu) is provided.

![Walker](https://github.com/eric-heiden/mocap-roboschool-demo/blob/doc/simple_walker.gif?raw=true)

## Dependencies
* [Roboschool](https://github.com/openai/roboschool)

## Run
```
python3 mocap_demo.py
```
