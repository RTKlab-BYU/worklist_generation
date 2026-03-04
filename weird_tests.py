import sys
import re

word = "banana"

newword = re.sub("a", "u", word)

print(newword)