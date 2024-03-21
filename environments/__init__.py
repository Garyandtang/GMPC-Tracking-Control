from gymnasium.envs.registration import register
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
register(
    id='turtlebot-v0',
    entry_point='environments.wheeled_mobile_robot.turtlebot.turtlebot:Turtlebot',
)