from pathlib import Path
from PIL import Image
import numpy as np

images = []
y_offset = 0
square_size = 16
save_image = None

# data_dir = Path("../PyTorch-VAE/Data/Minigrid")
data_dir = Path("../PyTorch-VAE/Data/Minigrid_Nohighlight")
imgs = sorted([f for f in data_dir.iterdir() if f.suffix == '.png'])

randomnumbers = np.random.randint(len(imgs), size=(square_size, square_size))

for i in range(square_size):
    images.append([])
    for j in range(square_size):
        images[i].append(Image.open(str(imgs[randomnumbers[i][j]])))

    widths, heights = zip(*(i.size for i in images[i]))

    total_width = sum(widths)
    max_height = max(heights)

    new_save_image = Image.new('RGB', (total_width, max_height*(i+1)))
    if save_image:
        new_save_image.paste(save_image, (0, 0))
    save_image = new_save_image

    x_offset = 0
    for im in images[i]:
        save_image.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
    y_offset += max_height
save_image.save('example.png')
