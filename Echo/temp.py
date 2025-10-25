# import matplotlib.pyplot as plt
# values1 = [0.61, 0.45, 0.23, 0.54, 0.86, 0.35, 0.02, 0.19, 0.77, 0.59, 0.4, 0.95]
# values2 = [0.76, 0.11, 0.08, 0.37, 0.89, 0.51, 0.0, 0.3, 0.88, 0.72, 0.29, 0.98]
# values4 = [0.72, 0.04, 0.25, 0.25, 1.0, 0.57, 0.0, 0.21, 0.83, 0.36, 0.74, 1.0]
# values8 = [0.73, 0.07, 0.4, 0.9, 1.0, 0.82, 0.0, 0.7, 0.88, 0.51, 0.52, 1.0]
#
# songs = ["Compensating", "LostInJapan", "LoveAndWar", "OtherBeds", "Swervin", "WhatDoYouMean",
#          "CaliforniaGurls", "ChuckTaylor", "CrazyTrain", "HowDeep", "Starlight", "TokyoDrift"]
# plt.figure(figsize=(20, 10))
# plt.plot(songs, values1, marker="o", linestyle="-", color="b", label="1")
# plt.plot(songs, values2, marker="o", linestyle="-", color="r", label="2")
# plt.plot(songs, values4, marker="o", linestyle="-", color="g", label="4")
# plt.plot(songs, values8, marker="o", linestyle="-", color="m", label="8")
# plt.ylim(0, 1.05)  # set y-axis range for clarity
# plt.title("Preciznost procene pesme")
# plt.xlabel("Pesme")
# plt.ylabel("Preciznost")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend()
# plt.savefig("VAR.png", dpi=300, bbox_inches="tight")
val=[0.73, 0.07, 0.4, 0.9, 1.0, 0.82, 0.0, 0.7, 0.88, 0.51, 0.52, 1.0]
print(sum(val)/len(val))