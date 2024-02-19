import matplotlib.pyplot as plt
import math

MAX_ITERATIONS = 2000
CWND = 83000

def sustained_load_tcp():
    windows = []
    i = 0
    window = 1
    time_to_first_congestion = None
    num_congestions = 0
    last_congested_at = None
    avg_time_between_congestions = 0

    while i < MAX_ITERATIONS:
        windows.append(window)
        ## Additive increase
        if window <= CWND:
            if window <= 38:
                alpha = 1
            else:
                ## Custom alpha function
                alpha = 500 / (1 + math.exp(-0.0001 * (window - 40000)))

            window += int(alpha)
        ## Multiplicative decrease
        else:
            ## Avg excludes the time taken to reach first convergence
            ## Only starts counting from first convergence onwards
            if last_congested_at != None:
                avg_time_between_congestions = (avg_time_between_congestions * num_congestions + (i - last_congested_at)) / (num_congestions + 1)
            
            last_congested_at = i
            num_congestions += 1

            if time_to_first_congestion == None:
                time_to_first_congestion = i

            if window <= 38:
                beta = 1
            else:
                ## Custom beta function
                beta = 2 / math.pow(window, 0.2)
            window = int(window * (1 - beta))

        i += 1
    
    return windows, time_to_first_congestion, num_congestions, avg_time_between_congestions

windows, time_to_first_congestion, num_congestions, avg_time_between_congestions = sustained_load_tcp()
print("=====SUSTAINED LOAD USER PROFILE=====")
print('Time to first congestion:', time_to_first_congestion / 10, 'seconds')
print('Number of congestions in', MAX_ITERATIONS, 'RTT:', num_congestions)
print('Average time between congestions:', avg_time_between_congestions / 10, 'seconds')

plt.plot([_ for _ in range(MAX_ITERATIONS)], windows, label="Window size (w)")
plt.xlabel("RTT")
plt.ylabel("W")
plt.title("W against RTT for Sustained Load User")
plt.legend()
plt.show()