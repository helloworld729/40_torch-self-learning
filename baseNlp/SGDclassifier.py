import numpy as np
from sklearn import linear_model
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 3])
clf = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
              eta0=0.0, fit_intercept=True, l1_ratio=0.15,
              learning_rate='optimal', loss='hinge', max_iter=50, n_jobs=1,
              penalty='l2', power_t=0.5, random_state=None, shuffle=True,
              verbose=0, warm_start=False)

clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
print(clf.coef_)