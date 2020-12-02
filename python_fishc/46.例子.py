class Myproperty:
	def __init__(self , fget = None , fset = None , fdel = None):
		self.fget = fget
		self.fset = fset
		self.fdel = fdel
	def __get__(self , instance , owner):
		return self.fget(instance)
	def __set__(self , instance , value):
		return self.fset(instance , value)
	def __delete__(self , instance):
		return self.fdel(instance)

class C:
	def __init__(self):
		self._x = None
	def getx(self):
		return self._x

	def setx(self , value):
		self._x = value
	def delx(self):
		del self._x
	x = Myproperty(getx , setx , delx)
