# 계층 각각을 클래스로 구현
# 모든 계층은 forward()와 backward()라는 메서드를 갖도록 구현

# 곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        return out
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# 덧셈 계층
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x+y
        return out
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
# 예시 구현
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num) # 100 * 2  = 200
price = mul_tax_layer.forward(apple_price, tax) # 200 * 1.1 = 220
print(price) # 220

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax) # 2.2 110 200
