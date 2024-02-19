import math
import random
import numpy as np
import matplotlib.pyplot as plt

SUSTAINED_LOAD_PROFILE = 'sustained_load'
BURST_LOAD_PROFILE = 'burst_load'
MAX_ITERATIONS = 1440000    # 1440000 RTT = 1440000 ms = 24 mins

class User:
    def __init__(self, profile, window = 1):
        self.profile = profile
        self.window = window

class DataCenter:
    def __init__(self, max_bandwidth, users=[], max_iterations=MAX_ITERATIONS):
        self.max_bandwidth = max_bandwidth
        self.users = users
        self.max_iterations = max_iterations
        self.packet_size = 1500 * 8   # 1500 bytes
        self.rtt = 0.1   # 100 ms
        self.congestion_window = self.compute_congestion_window()
        self.convergence_result = []    # Stores the convergence result in terms of cos(theta)
        self.slow_tcp_thres = 38  # Below 38 CWND we use standard TCP

    def compute_window_sum(self):
        return sum([user.window for user in self.users])
    
    def compute_congestion_window(self):
        return self.max_bandwidth * self.rtt / self.packet_size

    def alpha(self, user, users_window_sum):
        if user.profile == SUSTAINED_LOAD_PROFILE:
            return 500 / (1 + math.exp(-0.0001 * (users_window_sum - 40000)))
        else:
            return 50 * math.log(0.0005 * users_window_sum + 1) + 1
        
    def beta(self, user, users_window_sum):
        if user.profile == SUSTAINED_LOAD_PROFILE:
            return 2 / math.pow(users_window_sum, 0.2)
        else:
            return 50 / math.pow(users_window_sum + 10000, 0.5)
            
    def run(self):
        i = 0
        time_for_convergence = None
        while i < self.max_iterations:
            users_window_sum = self.compute_window_sum()

            ## Additive increase
            if users_window_sum <= self.congestion_window:
                if users_window_sum <= self.slow_tcp_thres:
                    for user in self.users:
                        user.window += 1
                else:
                    for user in self.users:
                        user.window = int(user.window + self.alpha(user, users_window_sum))
            
            ## Multiplicative decrease
            else:
                if users_window_sum <= self.slow_tcp_thres:
                    for user in self.users:
                        user.window = int(user.window * 0.5)
                else:
                    for user in self.users:
                        user.window = int(user.window * (1 - self.beta(user, users_window_sum)))

            converged = self.test_convergence(users_window_sum)
            if not time_for_convergence and converged:
                time_for_convergence = i
            # This is required to get the last convergence that did not fluctuate below 0.99
            elif not converged:
                time_for_convergence = None
            i += 1
        
        if converged:
            return time_for_convergence
        else:
            return None

    def test_convergence(self, users_window_sum):
        beta_matrix = [[0 for _ in range(len(self.users))] for _ in range(len(self.users))]
        alpha_matrix = [[0] for _ in range(len(self.users))]
        one_minus_beta_matrix = [[0 for _ in range(len(self.users))]]

        for i in range(len(self.users)):
            beta_matrix[i][i] = self.beta(self.users[i], users_window_sum)
            alpha_matrix[i][0] = self.alpha(self.users[i], users_window_sum)
            one_minus_beta_matrix[0][i] = 1 - self.beta(self.users[i], users_window_sum)

        sum_of_alphas = sum([alpha_matrix[i][0] for i in range(len(self.users))])

        A = beta_matrix + (1 / sum_of_alphas) * np.matmul(np.array(alpha_matrix), np.array(one_minus_beta_matrix))
        
        w_k = np.array([user.window for user in self.users])
        w_k_plus_1 = np.matmul(A, w_k)

        ## Test that w_k_plus_1 is parallel to w_k
        ## We use the cos(theta) formula to test for parallelism
        ## If the result is greater than 0.99, then the vectors are considered parallel
        self.convergence_result.append(np.dot(w_k_plus_1, w_k) / (np.linalg.norm(w_k_plus_1) * np.linalg.norm(w_k)))
        return np.dot(w_k_plus_1, w_k) / (np.linalg.norm(w_k_plus_1) * np.linalg.norm(w_k)) > 0.99


## Randomly generate n number of user profiles, 
## with varying chance of being sustained load and burst load
def generate_random_user_profiles(n):
    profiles = [SUSTAINED_LOAD_PROFILE, BURST_LOAD_PROFILE]
    users = []
    burst_load = 0
    sustained_load = 0
    for i in range(n):
        i = random.randint(0, 1)
        if i == 0:
            sustained_load += 1
        else:
            burst_load += 1
        users.append(User(profiles[i]))
    
    return users, burst_load, sustained_load

## Generate n number of user profiles,
## with k number of burst load profiles
## and n - k number of sustained load profiles
def generate_user_profiles(n, k):
    return [User(BURST_LOAD_PROFILE) for _ in range(k)] + [User(SUSTAINED_LOAD_PROFILE) for _ in range(n - k)]


def main():
    n = 2  ## Modify this data to test different number of users
    ## Modify the second parameter to test different number of burst load users
    users = generate_user_profiles(n, 1)    

    ## Generate random allocation of user profiles
    # users, burst_load, sustained_load = generate_random_user_profiles(n)
    # print('Burst load users:', burst_load)
    # print('Sustained load users:', sustained_load)

    # Data Center
    dataCenter = DataCenter(math.pow(10, 10), users)   # 10 Gbps
    # Start simulation
    res = dataCenter.run()

    # Sanity test
    if not res:
        print('Did not converge!')
        exit()


    time_for_convergence = res
    print('Time for convergence:', time_for_convergence / 10, 'seconds')
    print('Converged!') 

    # Print the right eigen vector
    print('Right Eigenvector:', np.array([user.window for user in users]))

    profiles = {SUSTAINED_LOAD_PROFILE: [], BURST_LOAD_PROFILE: []}
    for user in users:
        profiles[user.profile].append(user.window)
    
    # Fairness is taking the packet allocation for 1 SL profile and dividing it by the packet allocation for another BL profile
    if profiles[SUSTAINED_LOAD_PROFILE] and profiles[BURST_LOAD_PROFILE]:
        print('Fairness, f: ', min(profiles[SUSTAINED_LOAD_PROFILE]) / min(profiles[BURST_LOAD_PROFILE]))
    else:
        print('No fairness since only one profile is present')

    # Plot the convergence result
    # It should show a line eventually settling above 0.99
    plt.plot(dataCenter.convergence_result)
    plt.xlabel("RTT")
    plt.ylabel("cos(theta)")
    plt.title("Convergence Result")
    plt.show()

if __name__ == "__main__":
    main()