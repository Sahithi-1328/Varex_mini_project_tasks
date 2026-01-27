import numpy as np
import matplotlib.pyplot as plt
file_path = "../02_base.raw"

with open(file_path, "rb") as f:
    img_data = np.fromfile(f, dtype=np.uint16)

height = 3520
width = 4288
image = img_data.reshape((height, width))

row = int(input("enter row number: "))
col = int(input("enter column number: "))

image_value = image[row, col]
print(f"pixel at {row}, column {col} has value {image_value}")

image_norm = image / image.max()

plt.figure(figsize=(4, 5))
plt.imshow(image_norm, cmap="gray")
plt.scatter(col, row, c="red", s=50)
plt.text(col + 30, row + 30, f"({col}, {row})\n  ADU: {image_value}", color="red", fontsize=9)
plt.title("Detector image with selected pixel")
plt.axis("off")
plt.show()
