from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from generate_dataset import generate_normalized_uniform_2d

def svc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)

    clf = SVC()

    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))

if __name__ == '__main__':
    X, y = generate_normalized_uniform_2d(20000,0.20,5)
    print('--- TRAINING ---')
    svc(X,y)
