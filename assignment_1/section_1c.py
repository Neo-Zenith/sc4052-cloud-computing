import matplotlib.pyplot as plt
import math

MAX_ITERATIONS = 1440000
CWND = 83000

def diff_profile_exp():
    x1_values = []
    x2_values = []

    i = 0
    x1, x2 = 1, 1
    while i < MAX_ITERATIONS:
        x1_values.append(x1)
        x2_values.append(x2)
        if x1 + x2 <= CWND:
            if x1 + x2 <= 38:
                alpha1 = 1
                alpha2 = 1
            else:
                alpha1 = 50 * math.log(0.0005 * (x1 + x2) + 1) + 1
                alpha2 = 500 / (1 + math.exp(-0.0001 * (x1 + x2 - 40000)))

            x1 += int(alpha1)
            x2 += int(alpha2)
        else:
            if x1 + x2 <= 38:
                beta1 = 0.5
                beta2 = 0.5
            else:
                beta1 = 50 / math.pow(x1 + x2 + 10000, 0.5)
                beta2 = 2 / math.pow(x1 + x2, 0.2)

            x1 = int(x1 * (1 - beta1))
            x2 = int(x2 * (1 - beta2))
        i += 1
    
    return x1_values, x2_values

## Plot packet allocation between burst load user and sustained load user
x1_values, x2_values = diff_profile_exp()
plt.plot(x1_values, x2_values, label="Packet Allocation")

## Fairness line
plt.plot(range(CWND), range(CWND), label="Fairness", linestyle="--")

## Efficiency line
plt.plot([0, 83000], [83000, 0], label='Efficiency', linestyle='--')

plt.xlabel("Burst Load User")
plt.ylabel("Sustained Load User")

plt.title("Packet Allocation between Burst Load User and Sustained Load User")
plt.legend()

plt.show()


## Plot individual window size against RTT
plt.plot([_ for _ in range(2000)], x1_values[:2000], label="Burst Load User")
plt.plot([_ for _ in range(2000)], x2_values[:2000], label="Sustained Load User")

plt.xlabel("RTT")
plt.ylabel("Window Size")

plt.title("Window Size against RTT for Burst Load User and Sustained Load User")
plt.legend()

plt.show()