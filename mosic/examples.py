from draw import *
from run import *
import matplotlib.pyplot as plt


'''
To get started, you need to have dataset set up:
1) install wget, unzip and ffmpeg.
2) execute "python pre.py" to generate dataset,
   or do "python pre.py cache; python pre.py generate" if you want to keep the cache.
3) create some missing folders when required

The following code runs an experiment with inception modes using given args.
'''

runs(Inception, continuous=True, batchsize=8, resolution=2,
              spec='cqt', phase=False, epochs=100, size=256, lr=1e-4)

'''
The code below draws the loss verses time of the first experiment run for 100 epochs.
It also prints all the experiments run for 100 epochs.
'''

plt.style.use('bmh')
filtered = find(epochs=100)
fig, ax1 = plt.subplots(figsize=(4.5, 3))
plt.ylim((0.000, 0.015))
path = ('valid', 'gnr', 'loss')
plt.legend()
draw(ax1, filtered[0], ('valid', 'gnr', 'loss'), True, label='example', color='red')
plt.legend()
plt.ylabel(r'validation loss')
plt.xlabel('minutes')
plt.savefig('plots/example.svg')
plt.show()

show(filtered, by=['gnr-loss'])
