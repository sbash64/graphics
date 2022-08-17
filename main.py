from cmath import sin
from math import radians

print(('{{' + (89) * '{:6f}, ' + '{:6f}}}').format(*
      [sin(radians(x)).real for x in range(90)]))
