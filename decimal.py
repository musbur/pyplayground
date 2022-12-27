import decimal
from decimal import Decimal

ctx = decimal.getcontext()

def rd(x, prec):
    with decimal.localcontext() as context:
        context.prec = prec-1 if x.as_tuple()[1][0] > 5 else prec
        y = +Decimal(x) * Decimal('1.00000')
    return y


numbers = [Decimal('12').scaleb(e) for e in range(-4, 3)]

for n in numbers:
#    with decimal.localcontext() as c:
#        c.prec = 3
#        n = n * Decimal('1.00000')
    if n.to_integral_value().as_tuple() == n.as_tuple():
        n = n.quantize(Decimal(1))
    print(n)

with decimal.localcontext() as ctx:
    ctx.prec = 4
    x = ctx.create_decimal_from_float(0.2)
    y = ctx.create_decimal(10)
    print(y.quantize(1))


class MyDecimal(decimal.Decimal):
    def __str__(self):
        if self == self.to_integral():
            return super().quantize(Decimal(1)).__str__()
        return super().normalize().__str__()


m1 = MyDecimal('0.001')
m2 = MyDecimal('10.00')

print(m1)
print(m2)
