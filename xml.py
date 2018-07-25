# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2018 Luca Anzalone
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
# -- XML
# -----------------------------------------------------------------------------
import os


# constants:
header = """
<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>Training-set</name>
<comment>generated with imgann</comment>
<images>
"""
t1 = "  "
t2 = "    "
t3 = "      "


# -----------------------------------------------------------------------------
# -- Class
# -----------------------------------------------------------------------------
class Xml:
    def __init__(self, name, path="", mode="w"):
        self.path = os.path.join(path, name)
        self.file = open(self.path, mode)
        self.mode = mode

        if mode == "w":
            self.file.write(header)

    def append(self, path, boxes, points):
        # add a new entry: image, box, points
        f = self.file
        f.write(f"{t1}<image file='{path}'>\n")

        # write points for every box
        for i, box in enumerate(boxes):
            t, l, w, h = box

            s = f"{t2}<box top='{t}' left='{l}' width='{w}' height='{h}'>\n"
            f.write(s)

            for k, points in enumerate(points[i]):
                x, y = points
                f.write(f"{t3}<part name='{k}' x='{x}' y='{y}'/>\n")

            f.write(f"{t2}</box>\n")
        f.write(f"{t1}</image>\n")

    def close(self):
        if self.mode == "w":
            self.file.write("</images>\n")
            self.file.write("</dataset>\n")

        self.file.close()
# -----------------------------------------------------------------------------
