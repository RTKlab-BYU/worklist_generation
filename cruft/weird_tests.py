from math import sqrt

def isPrime(number):
    # assume the number is prime
    prime = True
    top = int(sqrt(number))
    for i in range(2, top + 1): # number = 9, 2-5
        if number % i == 0:
            prime = False
    return prime

start = int(input('Enter starting number: '))
end = int(input('Enter ending number: '))

primes = 0
non_primes = 0
for i in range(start, end +1):
    if isPrime(i):
        primes += 1
    else:
        non_primes += 1

print(f'In the range of {start} to {end}, there are {primes} prime numbers and {non_primes} non-prime numbers.')
