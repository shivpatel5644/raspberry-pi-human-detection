from gpiozero import LED
from time import sleep

green = LED(17)
yellow = LED(27)
red = LED(22)

while True:
    green.on()
    sleep(1)
    green.off()

    yellow.on()
    sleep(1)
    yellow.off()

    red.on()
    sleep(1)
    red.off()
