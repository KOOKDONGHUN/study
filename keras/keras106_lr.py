# 0.5를 넣고 0.8을 찾아간다

weight = 0.5
input = 0.5

goal_prediction = 0.0

lr = 0.001 # 0.1 /0.01 / 1

for iteration in range(1101):
    prediction = input * weight
    error = (prediction  - goal_prediction) #** 2

    print("Error : " + str(error) + "\tPrediction : ", str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) #** 2

    down_pred = input * (weight - lr)
    down_error = (goal_prediction - down_pred) #** 2

    if (down_error < up_error):
        weight = weight - lr

    if (down_error > up_error):
        weight = weight + lr
    