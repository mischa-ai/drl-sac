import numpy as np
from jetbot import Robot, Camera
import time
import Jetson.GPIO as GPIO
import cv2

class JetbotEnvironment:
    def __init__(self):
        # Jetbot setup
        self.robot = Robot()
        self.camera = Camera.instance(width=224, height=224)
        self.state_dim = 6  # Assuming 6 sensor inputs
        self.action_dim = 2  # Assuming 2 motor outputs (left and right motor speeds)

        # Set up the Jetson GPIO pins for the robot sensors (buttons)
        self.button_pins = [11, 12, 13, 15, 16, 19]
        GPIO.setmode(GPIO.BOARD)
        for pin in self.button_pins:
            GPIO.setup(pin, GPIO.IN)

    def read_button_states(button_pins):
        # Read the button states into a NumPy array
        button_states = np.array([GPIO.input(pin) for pin in button_pins], dtype=np.float32)
        return button_states

    def reset(self):
        self.robot.stop()  # Stop the robot motors
        time.sleep(1)  # Give the robot some time to come to a stop

        # Read the initial state from the sensors
        state = self.read_button_states(self.button_pins)

        return state

    def step(self, action):
        # Apply the action to the robot motors
        left_motor_speed = action[0]
        right_motor_speed = action[1]
        self.robot.set_motors(left_motor_speed, right_motor_speed)

        # Sleep for a small duration to allow the robot to execute the action
        time.sleep(0.1)  # You can adjust this duration based on your requirements

        # Read the next state from the sensors and the camera
        sensor_state = self.read_button_states()
        img_state = self.camera.value
        next_state = (img_state, sensor_state)

        # Calculate the reward based on the sensor readings
        # You can customize the reward calculation based on your specific problem requirements
        # In this example, we give a positive reward if no sensor is triggered, and a negative reward if any sensor is triggered
        if np.any(sensor_state == 1):  # Assuming a triggered sensor returns 1
            reward = -10
        else:
            reward = 1

        # Determine if the episode is done (e.g., if the robot has collided with an obstacle)
        # In this example, we consider the episode done if any sensor is triggered
        done = np.any(sensor_state == 1)

        return next_state, reward, done

    def close(self):
        self.robot.stop()
        self.camera.stop()
        GPIO.cleanup()
