import pandas as pd
import matplotlib.pyplot as plt

# 너의 결과
my_result = {
    'HOTA': 60.609, 'MOTA': 56.179, 'IDSW': 0, 'FP': 0, 'FN': 45,
    'MT': 0.0, 'ML': 44.94
}

# SOTA 예시 결과
models = ['FairMOT', 'ByteTrack', 'OC-SORT', 'YourModel']
results = {
    'HOTA': [63.2, 64.7, 65.9, my_result['HOTA']],
    'MOTA': [73.7, 80.3, 77.5, my_result['MOTA']],
    'IDSW': [1117, 564, 852, my_result['IDSW']],
    'FP': [27507, 19474, 19918, my_result['FP']],
    'FN': [117477, 93161, 98510, my_result['FN']],
    'MT': [35.3, 40.6, 36.2, my_result['MT']],
    'ML': [20.2, 19.3, 20.1, my_result['ML']],
}

df = pd.DataFrame(results, index=models)

# 시각화 예시 (HOTA 기준)
df[['HOTA', 'MOTA']].plot(kind='bar')
plt.title("Tracker Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
