import matplotlib.pyplot as plt

# Data
songs = ["Compensating", "LostInJapan", "LoveAndWar", "OtherBeds", "Swervin", "WhatDoYouMean",
         "CaliforniaGurls", "ChuckTaylor", "CrazyTrain", "HowDeep", "Starlight", "TokyoDrift"]

# Song = [PocetakPesme, Refren, Strofa(, Instrumental)] na 1sec srednji nivo suma
CaliforniaGurls1 = [1.0, 1.0, 0.99, 0]
CaliforniaGurls2 = [0, 0.13, 0.04, 0]
ChuckTaylor1 = [1.0, 1.0, 1.0, 0]
ChuckTaylor2 = [0.29, 0.18, 0.14, 0]
Compensating1 = [0.93, 1.0, 0.93, 1.0]
Compensating2 = [0.72, 0.81, 0.71, 0.31]
CrazyTrain1 = [1.0, 1.0, 1.0, 1.0]
CrazyTrain2 = [0.86, 1.0, 1.0, 1.0]
HowDeep1 = [1.0, 1.0, 1.0, 0]
HowDeep2 = [0.72, 0.83, 0.67, 0]
LostInJapan1 = [0.83, 1.0, 1.0, 0.84]
LostInJapan2 = [0.12, 0.71, 0.27, 0.08]
LoveAndWar1 = [1.0, 1.0, 1.0, 1.0]
LoveAndWar2 = [0.1, 0.65, 0, 1.0]
OtherBeds1 = [1.0, 1.0, 1.0, 0]
OtherBeds2 = [0.43, 0.63, 0.47, 0]
Starlight1 = [1.0, 1.0, 1.0, 0]
Starlight2 = [0.42, 0.51, 0.58, 0]
Swervin1 = [1.0, 1.0, 1.0, 1.0]
Swervin2 = [0.92, 0.85, 0.95, 1.0]
TokyoDrift1 = [1.0, 1.0, 1.0, 1.0]
TokyoDrift2 = [0.95, 0.96, 1.0, 1.0]
WhatDoYouMean1 = [0.91, 1.0, 1.0, 0.9]
WhatDoYouMean2 = [0.55, 1.0, 0.72, 0.0]
Segmenti = ["Početak pesme", "Refren", "Strofa", "Instrumental"]

# Duration 1
values01 = [0.95, 0.99, 1.0, 1.0, 1.0, 1.0, 0.99, 1.0, 1.0, 0.99, 1.0, 1.0]
values11 = [0.98, 0.74, 1.0, 1.0, 1.0, 0.98, 0.97, 1.0, 1.0, 0.99, 1.0, 1.0]
values21 = [0.61, 0.45, 0.23, 0.54, 0.86, 0.35, 0.02, 0.19, 0.77, 0.59, 0.4, 0.95]
values31 = [0.28, 0.35, 0.0, 0.0, 0.77, 0.22, 0.0, 0.0, 0.35, 0.14, 0.22, 0.0]

# Duration 2
values02 = [0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0]
values12 = [0.98, 0.81, 1.0, 1.0, 1.0, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
values22 = [0.68, 0.64, 0.68, 0.55, 0.81, 0.92, 0.1, 0.17, 1.0, 0.8, 0.68, 0.88]
values32 = [0.02, 0.77, 0.74, 0.0, 0.37, 0.09, 0.0, 0.0, 0.08, 0.0, 0.12, 0.0]

# Duration 4
values04 = [0.96, 0.94, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
values14 = [0.97, 0.66, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.97, 1.0, 1.0, 1.0]
values24 = [0.72, 0.04, 0.25, 0.25, 1.0, 0.57, 0.0, 0.21, 0.83, 0.36, 0.74, 1.0]
values34 = [0.0, 0.21, 0.0, 0.0, 0.34, 0.02, 0.0, 0.0, 0.68, 0.11, 0.12, 0.0]

# Duration 8
values08 = [1.0, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99, 1.0, 1.0]
values18 = [1.0, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
values28 = [0.73, 0.07, 0.4, 0.9, 1.0, 0.82, 0.0, 0.7, 0.88, 0.51, 0.52, 1.0]
values38 = [0.0, 0.24, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.73, 0.0, 0.27, 0.0]


# Lengths
# values21 = [0.61, 0.45, 0.23, 0.54, 0.86, 0.35, 0.02, 0.19, 0.77, 0.59, 0.4, 0.95] # 1s
# values22 = [0.76, 0.11, 0.08, 0.37, 0.89, 0.51, 0.0, 0.3, 0.88, 0.72, 0.29, 0.98] # 2s
# values24 = [0.72, 0.04, 0.25, 0.25, 1.0, 0.57, 0.0, 0.21, 0.83, 0.36, 0.74, 1.0] # 4s
# values28 = [0.73, 0.07, 0.4, 0.9, 1.0, 0.82, 0.0, 0.7, 0.88, 0.51, 0.52, 1.0] # 8s

# Plot
plt.figure(figsize=(20, 10))

# plt.plot(songs, values08, marker="o", linestyle="-", color="b", label="Bez šuma")
# plt.plot(songs, values18, marker="o", linestyle="-", color="r", label="Nizak šum")
# plt.plot(songs, values28, marker="o", linestyle="-", color="g", label="Srednji šum")
# plt.plot(songs, values38, marker="o", linestyle="-", color="m", label="Visok šum")

# plt.plot(songs, values21, marker="o", linestyle="-", color="b", label="1s")
# plt.plot(songs, values22, marker="o", linestyle="-", color="r", label="2s")
# plt.plot(songs, values24, marker="o", linestyle="-", color="g", label="4s")
# plt.plot(songs, values28, marker="o", linestyle="-", color="m", label="8s")

plt.plot(Segmenti, Swervin1, marker="o", linestyle="-", color="b", label="Nizak šum")
plt.plot(Segmenti, Swervin2, marker="o", linestyle="-", color="r", label="Srednji šum")

plt.ylim(0, 1.05)  # set y-axis range for clarity
# plt.title("What do you mean")
plt.xlabel("Segmenti")
plt.ylabel("Preciznost")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Save
plt.savefig("SegmentiSwe.png", dpi=300, bbox_inches="tight")

# Optional: show the plot
# plt.show()
