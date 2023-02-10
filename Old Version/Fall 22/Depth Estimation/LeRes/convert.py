#!/usr/bin/env python3

import glob
from PIL import Image, ImageMath

for file in sorted(glob.glob('./leresdepth/*.png')):
	im = Image.open(file)
	#im = ImageMath.eval('im/256', {'im':im}).convert('L')
	im.convert('RGB').save(file.replace('.png', '.jpg'))
print('Done.')