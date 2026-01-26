import numpy as np
import matplotlib.pyplot as plt
height=3520
width=4288
image_files = {
    350: "../1_exposed_350ms.raw",
    500: "../2_exposed_500ms.raw",
    1000: "../3_exposed_1000ms.raw",
    2000: "../4_exposed_2000ms.raw",
    3000: "../5_exposed_3000ms.raw",
    4000: "../6_exposed_4000ms.raw"
}
## 3.1 Reading multiple detector images 
images={}
for time_ms,file_path in image_files.items():
    with open(file_path,"rb") as f:
        img_data=np.fromfile(f,dtype=np.uint16)
    image=img_data.reshape((height,width))
    images[time_ms]=image
print("Images loaded:",images.keys())

## 3.2 calculating average pixel values and plotting them
integration_times=[]
average_values=[]
average_pixels={}
for time_ms,image in images.items():
    avg_value=image.mean()
    average_pixels[time_ms]=avg_value
    integration_times.append(time_ms)
    average_values.append(avg_value)
    print(f"Integration time: {time_ms} ms -> Average pixel value: {avg_value:.2f}")
plt.figure(figsize=(6, 4))
plt.plot(integration_times, average_values, marker="o")
plt.xlabel("Integration Time (ms)")
plt.ylabel("Average Pixel Value (ADU)")
plt.title("Average Pixel Value vs Integration Time")
plt.grid(True)
plt.show()

## 3.4 curve fitting
times=np.array(integration_times)
avg_pixels=np.array(average_values)
coeffs=np.polyfit(times,avg_pixels,deg=2)
t_smooth = np.linspace(min(times), max(times), 300)
poly_func = np.poly1d(coeffs)
offset_fit = poly_func(t_smooth)
plt.figure(figsize=(6,4))
plt.plot(times, avg_pixels, 'o', label="Measured data")
plt.plot(t_smooth, offset_fit, '-', label="Polynomial fit")
plt.xlabel("Integration Time (ms)")
plt.ylabel("Average Pixel Value")
plt.title("Offset vs Integration Time (Polynomial Fit)")
plt.legend()
plt.grid(True)
plt.show()


