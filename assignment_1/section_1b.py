import matplotlib.pyplot as plt
import math

MAX_ITERATIONS = 2000
CWND = 83000

def burst_load_tcp():
    windows = []
    i = 0
    window = 1
    time_to_first_congestion = None
    num_congestions = 0
    last_congested_at = None
    avg_time_between_congestions = 0

    while i < MAX_ITERATIONS:
        windows.append(window)
        if window <= CWND:
            if window <= 38:
                alpha = 1
            else:
                alpha = 50 * math.log(0.0005 * window + 1) + 1

            window += int(alpha)
        else:
            if last_congested_at != None:
                avg_time_between_congestions = (avg_time_between_congestions * num_congestions + (i - last_congested_at)) / (num_congestions + 1)
            
            last_congested_at = i
            num_congestions += 1

            if time_to_first_congestion == None:
                time_to_first_congestion = i

            if window <= 38:
                beta = 0.5
            else:
                beta = 50 / math.pow(window + 10000, 0.5)
            window = int(window * (1 - beta))

        i += 1
    
    return windows, time_to_first_congestion, num_congestions, avg_time_between_congestions

windows, time_to_first_congestion, num_congestions, avg_time_between_congestions = burst_load_tcp()
print("=====BURST LOAD USER PROFILE=====")
print('Time to first congestion:', time_to_first_congestion / 10, 'seconds')
print('Number of congestions in', MAX_ITERATIONS, 'RTT:', num_congestions)
print('Average time between congestions:', avg_time_between_congestions / 10, 'seconds')
plt.plot([_ for _ in range(MAX_ITERATIONS)], windows, label="Window size (w)")

plt.xlabel("RTT")
plt.ylabel("W")

plt.title("W against RTT for Burst Load User")
plt.legend()

plt.show()