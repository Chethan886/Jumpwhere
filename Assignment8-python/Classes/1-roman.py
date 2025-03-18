# 1) Write a python class to convert an integer into a roman numeral and viceversa

class RomanConverter:
    def __init__(self):
        self.roman_map = {
            1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL', 50: 'L', 
            90: 'XC', 100: 'C', 400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'
        }
        self.values = sorted(self.roman_map.keys(), reverse=True)

    def int_to_roman(self, num):
        roman = ""
        for value in self.values:
            while num >= value:
                roman += self.roman_map[value]
                num -= value
        return roman

    def roman_to_int(self, roman):
        roman_to_int_map = {v: k for k, v in self.roman_map.items()}
        i, num = 0, 0
        while i < len(roman):
            if i + 1 < len(roman) and roman[i:i+2] in roman_to_int_map:
                num += roman_to_int_map[roman[i:i+2]]
                i += 2
            else:
                num += roman_to_int_map[roman[i]]
                i += 1
        return num

converter = RomanConverter()
print(converter.int_to_roman(1994)) 
print(converter.roman_to_int("MCMXCIV")) 
