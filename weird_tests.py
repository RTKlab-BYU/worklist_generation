words = ['apple', 'banana', 'pear']
other_words = ['pear']
new = []
new.append(other_words[0])
words.remove(other_words[0])

print(new)
print(words)
print(other_words)