import numpy as np
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, SVC

from label_to_vector import get_XY


# 線形回帰
def train_lr(X_train2, Y_train2, X_test2, Y_test2):
  model = LogisticRegression(solver='newton-cg', max_iter=10000, multi_class='ovr')
  model.fit(X=X_train2, y=Y_train2)
  print_score(X_train2, Y_train2, X_test2, Y_test2, model)


# 線形SVM
def train_lsvc(X_train2, Y_train2, X_test, Y_test):
  model = LinearSVC(loss='hinge', C=1)
  model.fit(X=X_train2, y=Y_train2)
  print_score(X_train2, Y_train2, X_test, Y_test, model)


# 非線形SVM
def train_svc(X_train2, Y_train2, X_test, Y_test):
  model = SVC(kernel="rbf", gamma=5, C=0.001)
  model.fit(X=X_train2, y=Y_train2)
  print_score(X_train2, Y_train2, X_test, Y_test, model)


# ExtraTrees
def train_et(X_train2, Y_train2, X_test, Y_test):
  model = ExtraTreesClassifier(n_estimators=100)
  model.fit(X_train2, Y_train2)
  print_score(X_train2, Y_train2, X_test, Y_test, model)


def print_score(X_train2, Y_train2, X_test2, Y_test2, model):
  print("score: ", model.score(X=X_train2, y=Y_train2), accuracy_score(Y_test2, model.predict(X_test2)))


# 多層パーセプトロン
def train_dl(X_train, Y_train, X_test, Y_test):
  enc = OneHotEncoder(handle_unknown='ignore')
  enc.fit(np.array(Y_train).reshape(-1, 1))

  X_train2 = np.array(X_train)
  Y_train2 = enc.transform(np.array(Y_train).reshape(-1, 1)).toarray()
  assert len(X_train2) == len(Y_train2)
  X_test2 = np.array(X_test)
  Y_test2 = enc.transform(np.array(Y_test).reshape(-1, 1)).toarray()
  assert len(X_test2) == len(Y_test2)

  model = Sequential()
  model.add(Dense(64, activation='relu', input_shape=(X_train2.shape[1],)))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(Y_train2.shape[1], activation='softmax'))
  # model.summary()
  model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
  model.fit(X_train2, Y_train2, batch_size=64, epochs=100, verbose=1, validation_split=0.7)

  score = model.evaluate(X_test2, Y_test2, verbose=0)
  # print('Test loss:', score[0])
  print('Test accuracy', score[1])


if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = get_XY()
  train_lr(X_train, Y_train, X_test, Y_test)
  train_lsvc(X_train, Y_train, X_test, Y_test)
  train_svc(X_train, Y_train, X_test, Y_test)
  train_et(X_train, Y_train, X_test, Y_test)
  train_dl(X_train, Y_train, X_test, Y_test)
  pass
