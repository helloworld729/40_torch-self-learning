from tqdm import tqdm
from tqdm import trange
import time
from time import sleep
from random import random, randint

bar = tqdm(range(100), mininterval=2, leave=True, desc="hello")
for char in bar:
    bar.set_description("Processing %s" % char)
    bar.set_postfix(loss=random(), gen=randint(1, 999), str='详细信息', lst=[1, 2])
    sleep(0.1)
