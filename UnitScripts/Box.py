import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Enter the row of the image which has to be box plotted
sel = 204 
# Read from FrameSplit csv and get values which are needed - Filename and Coordinates
Frames = pd.read_csv('FramesSplit.csv')
Frames = Frames[['Framefile','Box']]
filename = Frames.iloc[sel,0]
coordinates = Frames.iloc[sel,1]
print(coordinates,filename)

coordinates = list(coordinates.replace('(','').replace(')','').split(','))
print (coordinates)
# Hardcoded Filename where the images are located corresponding to the Filename in FrameSplit csv
filename = "CV10_3/"+filename
im = Image.open(filename, mode='r')

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((int(coordinates[0]), int(coordinates[1])), int(coordinates[2]), int(coordinates[3]), linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
