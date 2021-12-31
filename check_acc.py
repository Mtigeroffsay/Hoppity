f=open("saves/cwe_full_JD_save/test/res.txt")
lines = f.readlines()
correct = 0
total = 0
for line in lines:
    r = line.split(" ")[1]
    total+=1
    correct+=int(r)

print(f'test size: %d' %total)
print(f'correct num: %d' %correct)
print(f'accuracy: %.2f' % (correct/total))

