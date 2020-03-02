import pandas
import random

dataset = pandas.read_csv("dataset.csv", header=None)

dataset = dataset.values

rowsD,columnsD=dataset.shape

X=dataset[:,0:columnsD-1]
rowsX,columnsX=X.shape

y=dataset[:,columnsD-1]

trainPercentage=0.7

X_train=X[0:int(0.7*rowsX),:]
y_train=y[0:int(0.7*rowsX)]

X_test=X[int(0.7*rowsX):rowsX,:]
y_test=y[int(0.7*rowsX):rowsX]

cp=[]

X_train_rows,X_train_columns=X_train.shape

for i in range(columnsX):
	cp.append({})
	for j in range(X_train_rows):
		if X_train[j,i] not in cp[i]:
			cp[i][X_train[j,i]]=[0,0]
		if y[j]=='No':
			cp[i][X_train[j,i]][0]+=1
		else:
			cp[i][X_train[j,i]][1]+=1

'''
for i  in cp:
	print(i)
'''

X_test_rows,X_test_columns=X_test.shape

hitcount=0
tp,tn=0,0
for i in y_train:
	if i=='No':
		tn+=1
	else:
		tp+=1

for i in range(X_test_rows):
	tpp=tp/(tp+tn)
	tnp=tn/(tp+tn)
	for j in range(columnsX):
		tnp*=(cp[j][X_test[i,j]][0]/tn)
		tpp*=(cp[j][X_test[i,j]][1]/tp)
		
	if (tpp>tnp and y_test[i]=='Yes') or (tnp>tpp and y_test[i]=='No'):
		hitcount+=1

print("Accuracy: ",hitcount*100/X_test_rows)