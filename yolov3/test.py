class Animal():
    def __init__(self,name,age) -> None:
        self.name = name
        self.age = 12

    def call(self):
        print(self.age)

class Cat(Animal):
    def __init__(self, name, age) -> None:
        super().__init__(name)

    def call(self):
        print(self.age)

if __name__=="__main__":
    a = Animal(1)
    a.call()