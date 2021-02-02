"""Play with some machine learning tutorial stuff.
https://www.youtube.com/watch?v=ok2s1vV9XW0"""

import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
print(dir(digits))  # Print list of attributes for object

# Display images
# plt.gray()
# for i in range(4):
#     plt.matshow(digits.images[i])
# plt.show()

# Print data
print(digits.data[:5])
df = pd.DataFrame(digits.data)
df.head()
df.show()
