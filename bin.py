import math
import sys

from MinZpoint import MinZPoint


class bin:
    has_point=False
    min_z=float('inf')
    min_z_range=0
    def addPoint(self,d,z):
        self.has_point=True
        if(z<self.min_z):
            self.min_z=z
            self.min_z_range=d
    def getMinZPoint(self):
        point =MinZPoint()
        if(self.has_point):
            point.z=self.min_z
            point.d=self.min_z_range
            return point

    def addPoint(self,point):
        d=math.sqrt(point.x*point.x+point.y*point.y)