from collections import Counter
from src.data_loader import load_test
from src.preprocessing import normalise_chars, is_noise_token
import re
cnt = Counter(w.lower() for x in load_test().text.head(100000)
              for w in re.findall(r"[A-Za-z]+", normalise_chars(x)) if is_noise_token(w))
print(cnt.most_common(60))
