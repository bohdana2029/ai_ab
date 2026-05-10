
'''
    if chr(keycode) == ' ':
      global paused
      paused = not paused
      print("Running" if not paused else "Paused")
  

last_modified = 0


while True:
    current_modified = os.path.getmtime(path_1)

    if current_modified != last_modified:
        model = mujoco.MjModel.from_xml_path(path_1)
        environment = mujoco.MjModel.from_xml_path(path_1)
        last_modified = current_modified
        print("Change detected, running simulation...")
        data_1 = mujoco.MjData(model)


        with mujoco.viewer.launch_passive(model,data_1,key_callback=key_callback)  as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < 15:
                step_start = time.time()
                #get a state of the environment, which would be the position and velocity of the drone, and the orientation of the drone
                #X = [𝐩, 𝐯, 𝝀, 𝜴,𝝎] T
                state = [data_1.qpos, data_1.qvel, data_1.xpos, data_1.cvel]
               # print(state)
                while viewer.is_running() and  time.time() - start < 10:
                    data_1.ctrl[0] = 90.0  
                    data_1.ctrl[1] = 70.0  
                    data_1.ctrl[2] = 90.0
                    data_1.ctrl[3] = 70.0
                    mujoco.mj_step(model, data_1)
                    print ('Position of the drone:', data_1.geom_xpos[id])
                    viewer.sync()
                data_1.ctrl[1] = 88.0
                data_1.ctrl[3] = 88.0 
                data_1.ctrl[0] = 88.0
                print ('Position of the drone:', data_1.geom_xpos[id])
                mujoco.mj_step( model, data_1)
    
                viewer.sync()
       

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    time.sleep(1)


     self.model = mujoco.MjModel.from_xml_path(path_1)
        self.data = mujoco.MjData(self.model)
        self.id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'drone')
        self.thurst1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'thurst1')
        self.thurst2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'thurst2')
        self.thurst3 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'thurst3')
        self.thurst4 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'thurst
    #
    # 
def key_callback(keycode):
    if chr(keycode) == ' ':
      global paused
      paused = not paused
      print("Running" if not paused else "Paused")

    # 
    # 
    # 
    # '''