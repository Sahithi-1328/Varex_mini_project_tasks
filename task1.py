import numpy as np
import matplotlib.pyplot as plt
file_path = "../02_base.raw"
with open(file_path, "rb") as f:
    # f.seek(2048)  # skip 1024 words × 2 bytes
    img_data = np.fromfile(f, dtype=np.uint16)
# size of img_data 
print("Pixels read:", img_data.size)
print(img_data.shape)
height=3520
width=4288
image=img_data.reshape((height,width))
# size of image after reshape
print(image.shape)
print("Min pixel:", image.min())
print("Max pixel:", image.max())
# normalization to keep values between 0.0 to 1.0
image_norm=(image / image.max())
plt.figure(figsize=(6,8))
plt.imshow(image_norm,cmap="gray",vmin=image.min(),vmax=image.max())
plt.axis("off")
plt.title("Detector image")
plt.show()
# basic image processing techniques
# 1. Histogram
plt.figure(figsize=(6,4))
plt.hist(image.flatten(),bins=100,color="gray")
plt.xlabel("Pixel values")
plt.ylabel("Number of pixels")
plt.title("Histogram of detector pixel values")
plt.show()
# 2. Contrast stretching min max
image_cs=((image-image.min())/image.max()-image.min())
plt.figure(figsize=(6,4))
plt.imshow(image_cs,cmap="gray")
plt.title("Contrast stretching")
plt.axis("off")
plt.show()
# 2. contrast stretching percentile based
p_low, p_high = np.percentile(image, (1, 99))
print("1st percentile:", p_low)
print("99th percentile:", p_high)
image_ps = (image - p_low) / (p_high - p_low)
plt.figure(figsize=(6,8))
plt.imshow(image_ps, cmap="gray", vmin=0, vmax=1)
plt.title("Contrast Stretching (Percentile-based)")
plt.axis("off")
plt.show()
# 3. Region of interest (ROI)
image_roi=image[2900:3100,4100:4287]
plt.figure(figsize=(5,5))
plt.imshow(image_roi,cmap="gray",
           vmin=image.min(),vmax=image.max())
plt.title("Region of interest")
plt.axis("off")
plt.show()
# 4. Subtracting Mean / baseline
mean_val=image.mean()
image_sub_mean=image-mean_val
plt.imshow(image_sub_mean,cmap="gray")
plt.title("Mean subtracted image")
plt.colorbar(label="Deviation from Mean")
plt.show()
# 5. Normalization
abs_max = np.percentile(np.abs(image_sub_mean), 99)
plt.figure(figsize=(6,8))
plt.imshow(image_sub_mean,cmap="gray",vmin=-abs_max,vmax= abs_max)
plt.colorbar(label="Deviation from Mean")
plt.title("Mean-Subtracted Image (Properly Scaled)")
plt.axis("off")
plt.show()




 

