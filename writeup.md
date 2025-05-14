# Problem (unicode1): Understanding Unicode (1 point)

(a) What Unicode character does chr(0) return?
Answer: \x00

(b) How does this character’s string representation (__repr__()) differ from its printed representa-
tion?
__repr__() produces the representation of the string, it wraps the character in quotes and shows
the byte/Unicode escape \x00, i.e. "'\x00'". print() emits the actual character, because \x00
is a control character with no visual glyph, nothing appears on-screen.

(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
In string representation it will be displayed as \x00, and will be no symbol when printed.
