from model.model import TDR


 
model = TDR(128, 231666, 4096, 64,256, 8, 500000)

parameters = [p for p in model.parameters()]
print(len(parameters))
