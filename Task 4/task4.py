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

# Step 2: Row-wise average pixel values (3 regions)

row_regions = {
    "R1": (0, 1428),
    "R2": (1429, 2857),
    "R3": (2858, 3519)
}

region_averages = {}

for time_ms, image in images.items():
    print(f"\nIntegration Time: {time_ms} ms")
    region_averages[time_ms] = {}

    for region, (r_start, r_end) in row_regions.items():
        region_pixels = image[r_start:r_end+1, :]
        avg_value = region_pixels.mean()

        region_averages[time_ms][region] = avg_value
        print(f"  {region} (rows {r_start}-{r_end}) → Avg = {avg_value:.2f}")

#  Create 3 plots (Region-wise average vs Integration Time)

integration_times=[]
region1_avg=[]
region2_avg=[]
region3_avg=[]
for time_ms,image in images.items():
    integration_times.append(time_ms)
    avg_r1=image[0:1429 ,:].mean()
    avg_r2=image[1429:2858 ,:].mean()
    avg_r3=image[2858:4288,:].mean()
    region1_avg.append(avg_r1)
    region2_avg.append(avg_r2)
    region3_avg.append(avg_r2)
    print(f"{time_ms} ms -> R1: {avg_r1:.2f}, R2: {avg_r2:.2f}, R3: {avg_r3:.2f}")
plt.figure(figsize=(6,4))
plt.plot(integration_times, region1_avg, marker='o')
plt.xlabel("Integration Time (ms)")
plt.ylabel("Average Pixel Value (ADU)")
plt.title("Region 1 (Rows 0–1428)")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(integration_times, region2_avg, marker='o')
plt.xlabel("Integration Time (ms)")
plt.ylabel("Average Pixel Value (ADU)")
plt.title("Region 2 (Rows 1429–2857)")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(integration_times, region3_avg, marker='o')
plt.xlabel("Integration Time (ms)")
plt.ylabel("Average Pixel Value (ADU)")
plt.title("Region 3 (Rows 2858–4287)")
plt.grid(True)
plt.show()

# 4.3 polynomial fit

times = np.array(integration_times)

r1 = np.array(region1_avg)
r2 = np.array(region2_avg)
r3 = np.array(region3_avg)

# Polynomial degree (low degree to avoid overfitting)
degree = 2

# Fit polynomials for each region
coeffs_r1 = np.polyfit(times, r1, deg=degree)
coeffs_r2 = np.polyfit(times, r2, deg=degree)
coeffs_r3 = np.polyfit(times, r3, deg=degree)

# Create polynomial functions
poly_r1 = np.poly1d(coeffs_r1)
poly_r2 = np.poly1d(coeffs_r2)
poly_r3 = np.poly1d(coeffs_r3)

# Print polynomial equations (for verification)
print("Region 1 polynomial:", poly_r1)
print("Region 2 polynomial:", poly_r2)
print("Region 3 polynomial:", poly_r3)

# Create smooth time axis for plotting fits
t_smooth = np.linspace(times.min(), times.max(), 300)

# -------- Plot Region 1 --------
plt.figure(figsize=(6,4))
plt.plot(times, r1, 'o', label="Region 1 data")
plt.plot(t_smooth, poly_r1(t_smooth), '-', label="Region 1 fit")
plt.xlabel("Integration Time (ms)")
plt.ylabel("Average Pixel Value (ADU)")
plt.title("Region 1 Polynomial Fit (Rows 0–1428)")
plt.legend()
plt.grid(True)
plt.show()

# -------- Plot Region 2 --------
plt.figure(figsize=(6,4))
plt.plot(times, r2, 'o', label="Region 2 data")
plt.plot(t_smooth, poly_r2(t_smooth), '-', label="Region 2 fit")
plt.xlabel("Integration Time (ms)")
plt.ylabel("Average Pixel Value (ADU)")
plt.title("Region 2 Polynomial Fit (Rows 1429–2857)")
plt.legend()
plt.grid(True)
plt.show()

# -------- Plot Region 3 --------
plt.figure(figsize=(6,4))
plt.plot(times, r3, 'o', label="Region 3 data")
plt.plot(t_smooth, poly_r3(t_smooth), '-', label="Region 3 fit")
plt.xlabel("Integration Time (ms)")
plt.ylabel("Average Pixel Value (ADU)")
plt.title("Region 3 Polynomial Fit (Rows 2858–4287)")
plt.legend()
plt.grid(True)
plt.show()

# 4.4 scaling images
ref_time = 500

# Reference offsets at 500 ms
ref_r1 = poly_r1(ref_time)
ref_r2 = poly_r2(ref_time)
ref_r3 = poly_r3(ref_time)

scaled_images_task4 = {}

for time_ms, image in images.items():

    if time_ms == ref_time:
        continue  # skip reference image

    # Compute scale factors for each region
    scale_r1 = ref_r1 / poly_r1(time_ms)
    scale_r2 = ref_r2 / poly_r2(time_ms)
    scale_r3 = ref_r3 / poly_r3(time_ms)

    # Convert image to float for safe scaling
    scaled_image = image.astype(np.float32).copy()

    # Apply region-wise scaling
    scaled_image[0:1429, :] *= scale_r1
    scaled_image[1429:2858, :] *= scale_r2
    scaled_image[2858:4288, :] *= scale_r3
    scaled_images_task4[time_ms] = scaled_image
    output_path = rf"C:\Users\sa995420\OneDrive - Varex Imaging\Documents\varex_zip_files for tasks\tasks\Task4_outputs\task 4.4\Scaled_images\scaled_{time_ms}ms.raw"
    scaled_image.astype(np.uint16).tofile(output_path)
    print(f"Task 4.4: Scaled image saved for {time_ms} ms")
 
 # validation plot

# Choose one scaled image to validate
time_to_check = int(input())
scaled_image = scaled_images_task4[time_to_check]

# Calculate average pixel value for each row
row_means = scaled_image.mean(axis=1)

# Plot row-wise average
plt.figure(figsize=(8,4))
plt.plot(row_means)
plt.xlabel("Row Number")
plt.ylabel("Average Pixel Value (ADU)")
plt.title(f"Row-wise Average After Region-wise Scaling ({time_to_check} ms)")
plt.grid(True)
plt.show()
