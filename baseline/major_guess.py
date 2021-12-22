acc = 0
target = []
B_n = 0
I_n = 0

for line in open('../test.txt'):
    words = line.split()
    if len(words) == 0:
        continue
    target.append(words[2][0])

# Since the majority is the label B, always guess B
for i in range(len(target)):
    if target[i] == 'B':
        acc += 1
        B_n += 1
    elif target[i] == 'I':
        I_n += 1

print("Tag accuracy:", acc / (B_n + I_n) * 100, "%")

# Since TN = B / total, FN = I / total, TP = 0, FP = 0
# Apply L'Hopital's Rule
precision = I_n / (B_n + I_n)
recall = 0
fscore = 2 * precision * recall / (precision + recall)

print("F1 score:", fscore * 100, "%")