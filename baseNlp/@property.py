class Student(object):
    def __int__(self):
        self._score = 80

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError("必须是int类型")
        if value < 0 :
            raise ValueError("out of boundary")
        self._score = value

if __name__ == '__main__':
    Knight = Student()
    Knight.score = 95
    print(Knight.score)  # 95
