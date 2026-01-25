import numpy as np
import matplotlib.pyplot as plt
file_path="../02_base.raw"
with open (file_path,"rb") as f:
    img_data=np.fromfile(f,dtype=np.uint16)
height=3520
width=4288
image=img_data.reshape((height,width))
row=int(input("Enter row number:"))
col=int(input("Enter column number:"))
image_value=image[row,col]
print(f"pixel at {row} , column {col} has value {image_value}")
image_norm=(image/image.max())
plt.figure(figsize=(10,14))
plt.imshow(image_norm,cmap="gray")
plt.scatter(col,row,c="red",s=40)
plt.text(col + 30, row + 30,f"({col}, {row})\nADU: {image_value}",color="red",fontsize=13)
plt.title("Detector image with selected pixel")
plt.show()