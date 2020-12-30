from HiParTIPy.PtiCffi import pti,PTI

class ValueVector:
    def __init__(self, length):
        self.address=pti.new("ptiValueVector *")
        PTI.ptiNewValueVector(self.address,length,length)

    # This method sets a value to a specific location
    # @:param i, integer of the index,
    # @:param value, number to place in
    def setValue(self, i,value):
        self.address.data[i]=value

    # This method writes the vector to a file
    # @:param fileName, string of the file's path
    def dumpVec(self, fileName):
        file=PTI.fopen(bytes(fileName,'ascii'),b"w")
        PTI.ptiDumpValueVector(self.address,file)

    # Fills the matrix with random values for testing
    def makeRandom(self):
        PTI.ptiRandomValueVector(self.address)

    # These get methods are to make an easy way for a user to get the instances of the structures in C
    # A user can still call self.address.<>, but we want to avoid that if possible.
    def get(self,i): return self.address.data[i]

class IndexVector:
    def __init__(self,length):
        self.address=pti.new("ptiNnzIndexVector *")
        PTI.ptiNewNnzIndexVector(self.address,length,length)

    # This sets the value in the vector
    # @:param i, integer value of the index to change
    # @:param value, integer value of the item placed at index
    def setValue(self, i, value):
        self.address.data[i]=value

    # This method quick sorts the integers in the array.
    def sort(self):
        PTI.ptiQuickSortNnzIndexArray(self.address.data,0,self.address.len)