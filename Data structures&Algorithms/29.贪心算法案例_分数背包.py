# 小偷装包,利润最大化,商品为金砂(可以细分)--分数背包
goods = [(120, 30), (60, 10), (100, 20)]                    # 每个商品价格,重量
goods.sort(key=lambda x: x[0] / x[1], reverse=True)         # 自定义排序,单价从大到小
print(goods)

def fractional_backpack(goods, w):
    m = [0 for _ in range(len(goods))]                      # 拿法
    total_price = 0
    for idx, (price, weight) in enumerate(goods):
        if w >= weight:
            w -= weight
            total_price += price
            m[idx] = 1
        else:                                               # 装满了
            total_price += w * price / weight
            m[idx] = round(w / weight, 3)                   # 保留三位小数
            break
    return total_price, m

print(fractional_backpack(goods, 35))


