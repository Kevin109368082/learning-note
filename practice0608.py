# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 23:47:19 2021

@author: Kevin
"""

import turtle

# Create screen and turtle objects
screen = turtle.Screen()
screen.setup(500, 400)
myTurtle = turtle.Turtle()

# Move the turtle
myTurtle.forward(150)
myTurtle.left(90)
myTurtle.forward(75)

# Exit
screen.exitonclick()