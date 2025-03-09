import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pandas as pd


def normalizer(df : np.ndarray):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

class FCMeans:
    def __init__(self, n_clusters=3, max_iter=100, m=2, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m  # fuzziness parameter
        self.tol = tol
        self.random_state = random_state
        self.centers = None
        self.u = None  # membership matrix

    def fit(self, X):
        # Set the random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        N = X.shape[0]
        
        # Initialize membership matrix with random values between 0 and 1
        self.u = np.random.rand(N, self.n_clusters)
        
        # Normalize so that sum of memberships for each data point equals 1
        self.u = self.u / np.sum(self.u, axis=1)[:, np.newaxis]

        for _ in range(self.max_iter):
            # Raise membership matrix to the power of m (fuzzification)
            u_m = self.u ** self.m
            # Store old centers to check for convergence
            old_centers = self.centers if self.centers is not None else None
            # Compute new cluster centers using weighted average of data points
            self.centers = (u_m.T @ X) / np.sum(u_m, axis=0)[:, np.newaxis]

            # Update membership values
            distances = np.zeros((N, self.n_clusters))
            for i in range(self.n_clusters):
                # Calculate the distance between each data point and the center of the cluster
                distances[:, i] = np.linalg.norm(X - self.centers[i], axis=1)   # Euclidean distance    
            
            # Avoid division by zero
            distances = np.fmax(distances, np.finfo(float).eps)
            
            # Calculate the membership values
            exp = 2./(self.m-1)
            # Inverse distance relation for membership calculation
            tmp = (distances ** (-exp))
            
            # Normalize the membership values
            self.u = tmp / np.sum(tmp, axis=1)[:, np.newaxis]

            # Check convergence
            if old_centers is not None:
                # Check if the centers have converged
                if np.all(np.abs(old_centers - self.centers) < self.tol):
                    break

class T1FRBS:
    """
    Type-1 Fuzzy Rule-Based System (zero-order TSK) for multidimensional input.
    
    Attributes:
        n_rules (int): Number of fuzzy rules.
        n_features (int): Number of input features.
        centers (np.array): Array of rule centers of shape (n_rules, n_features).
        sigma (np.array): Standard deviations for each input dimension.
        p (np.array): Zero-order TSK consequent parameters (constants) for each rule.
    """
    def __init__(self, n_rules, n_features, centers=None):
        self.n_rules = n_rules
        self.n_features = n_features
        self.centers = centers
        self.sigma = None  # array of shape (n_features,)
        self.p = np.zeros(n_rules)

    def compute_membership(self, x):
        """
        Compute the firing strength for each rule using product t-norm
        Optimized using NumPy vectorized operations
        """
        # Reshape x to 2D array for vectorized operations
        x = np.array(x).reshape(1, -1)
        
        # Calculate Gaussian membership functions for all features and rules vectorially
        diff = x[:, np.newaxis, :] - self.centers
        exp_term = -0.5 * (diff / self.sigma) ** 2
        memberships = np.exp(exp_term)
        
        # Apply product t-norm using prod along last axis
        mu = np.prod(memberships, axis=2).flatten()
        
        return mu

    def predict(self, x):
        """
        Predict output using weighted average defuzzification.
        """ 
        # Compute the firing strength for each rule
        mu = self.compute_membership(x)
        
        # Check if the sum of the membership values is zero
        if np.sum(mu) == 0:
            return 0
        
        # Compute the output using the weighted average defuzzification
        return np.sum(mu * self.p) / np.sum(mu)

    def fit(self, x_train, y_train):
        """
        Fit the zero-order TSK consequents using the training data.
        """
        N, d = x_train.shape
        
        # If the centers are not provided, use FCMeans to cluster the data
        if self.centers is None:
            fcm = FCMeans(n_clusters=self.n_rules)
            fcm.fit(x_train)
            self.centers = fcm.centers
        
        # Compute sigma for each dimension
        self.sigma = np.zeros(self.n_features)
        for j in range(self.n_features):
            # Compute the standard deviation for each dimension
            x_min = np.min(x_train[:, j])
            x_max = np.max(x_train[:, j])
            self.sigma[j] = (x_max - x_min) / (2 * self.n_rules)
        
        # Learn the consequent parameters
        self.p = np.zeros(self.n_rules)
        for i in range(self.n_rules):
            # Compute the weights for each rule
            weights = []
            for x in x_train:
                weights.append(self.compute_membership(x)[i])
            weights = np.array(weights)
            
            # Check if the sum of the weights is greater than zero
            if np.sum(weights) > 0:
                # Compute the consequent parameter
                self.p[i] = np.sum(weights * y_train) / np.sum(weights)
            else:
                # If the sum of the weights is zero, set the consequent parameter to zero
                self.p[i] = 0

    def predict_all(self, x_array):
        """
        Predict outputs for an array of input vectors.
        """
        # Predict the output for each input vector
        return np.array([self.predict(x) for x in x_array])

    def plot_rules(self, name, dim=0, num_points=100):
        """
        Plot the Gaussian membership functions for each rule.
        """
        if self.sigma is None:
            raise Exception("Model not fitted yet")
        
        c_vals = self.centers[:, dim]
        x_min_dim = np.min(c_vals) - 1
        x_max_dim = np.max(c_vals) + 1
        x_range = np.linspace(x_min_dim, x_max_dim, num_points)
        
        n_cols = math.ceil(math.sqrt(self.n_rules))
        n_rows = math.ceil(self.n_rules / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if self.n_rules == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(self.n_rules):
            ax = axes[i]
            center = self.centers[i, dim]
            mu_vals = gaussian_mf(x_range, center, self.sigma[dim])
            
            ax.plot(x_range, mu_vals, '-', label='Membership')
            ax.set_title(f'Rule {i+1}')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
        
        for j in range(self.n_rules, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{name}-rules.png")

def gaussian_mf(x, c, sigma):
    """
    Gaussian membership function.
    """
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def plot_predictions(y_true, y_pred, title):
    """
    Plot predicted versus true output values.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, color='blue', marker='o')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_fuzzy_sets(feature_name, membership_functions, x_range=(0, 100), num_points=1000):
    """
    Plot fuzzy sets for a given feature
    
    Parameters:
    - feature_name: name of the feature being plotted
    - membership_functions: dict of {set_name: membership_function}
    - x_range: tuple of (min, max) values for x-axis
    - num_points: number of points to plot
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    plt.figure(figsize=(10, 6))
    for set_name, mf in membership_functions.items():
        y = [mf(xi) for xi in x]
        plt.plot(x, y, label=set_name)
    
    plt.title(f'Fuzzy Sets for {feature_name}')
    plt.xlabel('Input Value')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    
    # Load the dataset from the Excel file
    df = pd.read_excel("concrete+slump+data.xlsx")


    # Pre-processing Steps

    # 1. Handle missing values (if any)
    df.dropna(inplace=True)         # Drop rows with missing values
    df.reset_index(drop=True, inplace=True)  # Reset the index after dropping rows

    # 2. Remove duplicate rows (if present)
    df.drop_duplicates(inplace=True)

    # 3. Optionally, scale the input features using StandardScaler (or another scaler)
    input_features = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]
    X = df[input_features].values
    X_normalize = normalizer(X)

    # 4. Extract output variables (targets)
    y_slump = df["SLUMP(cm)"].values
    y_flow  = df["FLOW(cm)"].values
    y_cs    = df["Compressive Strength (28-day)(Mpa)"].values 

    n_features = X.shape[1]
    n_rules = 10
    n_runs = 50   # Number of random splits to evaluate

    # Lists to collect RMSE results for each output over multiple runs.
    rmse_slump_list = []
    rmse_flow_list = []
    rmse_cs_list = []

    for run in range(n_runs):

        # 2. Randomly split the dataset (80% train, 20% test)
        indices = np.random.permutation(len(X_normalize))
        n_train = int(0.8 * len(X_normalize))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        X_train = X_normalize[train_idx]
        X_test = X_normalize[test_idx]
        
        # 3. Train one FRBS per output variable.
        # Slump model
        frbs_slump = T1FRBS(n_rules, n_features)
        frbs_slump.fit(X_train, y_slump[train_idx])
        y_pred_slump = frbs_slump.predict_all(X_test)
        rmse_slump = rmse(y_slump[test_idx], y_pred_slump)
        rmse_slump_list.append(rmse_slump)
        
        # Flow model
        frbs_flow = T1FRBS(n_rules, n_features)
        frbs_flow.fit(X_train, y_flow[train_idx])
        y_pred_flow = frbs_flow.predict_all(X_test)
        rmse_flow_val = rmse(y_flow[test_idx], y_pred_flow)
        rmse_flow_list.append(rmse_flow_val)
        
        # 28-day Compressive Strength model
        frbs_cs = T1FRBS(n_rules, n_features)
        frbs_cs.fit(X_train, y_cs[train_idx])
        y_pred_cs = frbs_cs.predict_all(X_test)
        rmse_cs_val = rmse(y_cs[test_idx], y_pred_cs)
        rmse_cs_list.append(rmse_cs_val)
        
    
    # 4. Report the best (lowest) and mean RMSE for each output variable.
    print("Results over {} runs:".format(n_runs))
    print("Slump: Mean RMSE = {:.4f}, Best RMSE = {:.4f}".format(np.mean(rmse_slump_list), np.min(rmse_slump_list)))
    print("Flow: Mean RMSE = {:.4f}, Best RMSE = {:.4f}".format(np.mean(rmse_flow_list), np.min(rmse_flow_list)))
    print("28-day Compressive Strength: Mean RMSE = {:.4f}, Best RMSE = {:.4f}".format(np.mean(rmse_cs_list), np.min(rmse_cs_list)))
    
    # 5. Optionally, plot predictions from one run (for example, the last run)
    plot_predictions(y_slump[test_idx], y_pred_slump, "Slump: True vs Predicted")
    plot_predictions(y_flow[test_idx], y_pred_flow, "Flow: True vs Predicted")
    plot_predictions(y_cs[test_idx], y_pred_cs, "28-day Compressive Strength: True vs Predicted")

    # 6. Optionally, plot membership functions for a selected input dimension (e.g., dimension 0)
    frbs_slump.plot_rules("Slump")
    frbs_cs.plot_rules("Compressive Strength")
    frbs_flow.plot_rules("Flow")

    
    
    # تعریف توابع عضویت برای هر ویژگی
    def create_membership_functions(feature_min, feature_max):
        sigma = (feature_max - feature_min) / 6  # برای پوشش مناسب دامنه
        c_low = feature_min + sigma
        c_medium = (feature_min + feature_max) / 2
        c_high = feature_max - sigma
        
        def low(x): return np.exp(-((x - c_low) ** 2) / (2 * sigma ** 2))
        def medium(x): return np.exp(-((x - c_medium) ** 2) / (2 * sigma ** 2))
        def high(x): return np.exp(-((x - c_high) ** 2) / (2 * sigma ** 2))
        
        return {'Low': low, 'Medium': medium, 'High': high}

    # برای هر ویژگی ورودی
    for feature in input_features:
        feature_min = df[feature].min()
        feature_max = df[feature].max()
        mfs = create_membership_functions(feature_min, feature_max)
        plot_fuzzy_sets(feature, mfs, x_range=(feature_min, feature_max))

if __name__ == "__main__":
    main()

def plot_fuzzy_sets(feature_name, membership_functions, x_range=(0, 100), num_points=1000):
    """
    Plot fuzzy sets for a given feature
    
    Parameters:
    - feature_name: name of the feature being plotted
    - membership_functions: dict of {set_name: membership_function}
    - x_range: tuple of (min, max) values for x-axis
    - num_points: number of points to plot
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    plt.figure(figsize=(10, 6))
    for set_name, mf in membership_functions.items():
        y = [mf(xi) for xi in x]
        plt.plot(x, y, label=set_name)
    
    plt.title(f'Fuzzy Sets for {feature_name}')
    plt.xlabel('Input Value')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# Assuming you have membership functions like these:
def low(x): return max(0, min(1, (20-x)/20)) if x <= 20 else 0
def medium(x): return max(0, min((x-10)/20, (50-x)/20)) if 10 <= x <= 50 else 0
def high(x): return max(0, min((x-40)/20, 1)) if x >= 40 else 0

membership_functions = {
    'Low': low,
    'Medium': medium,
    'High': high
}









# etc.
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import math
# import pandas as pd


# def normalizer(df : np.ndarray):
#     normalized_df = (df - df.min()) / (df.max() - df.min())
#     return normalized_df

# class FCMeans:
#     def __init__(self, n_clusters=3, max_iter=100, m=2, tol=1e-4, random_state=None):
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.m = m  # fuzziness parameter
#         self.tol = tol
#         self.random_state = random_state
#         self.centers = None
#         self.u = None  # membership matrix

#     def fit(self, X):
#         # Set the random seed if provided
#         if self.random_state is not None:
#             np.random.seed(self.random_state)

#         N = X.shape[0]
        
#         # Initialize membership matrix with random values between 0 and 1
#         self.u = np.random.rand(N, self.n_clusters)
        
#         # Normalize so that sum of memberships for each data point equals 1
#         self.u = self.u / np.sum(self.u, axis=1)[:, np.newaxis]

#         for _ in range(self.max_iter):
#             # Raise membership matrix to the power of m (fuzzification)
#             u_m = self.u ** self.m
#             # Store old centers to check for convergence
#             old_centers = self.centers if self.centers is not None else None
#             # Compute new cluster centers using weighted average of data points
#             self.centers = (u_m.T @ X) / np.sum(u_m, axis=0)[:, np.newaxis]

#             # Update membership values
#             distances = np.zeros((N, self.n_clusters))
#             for i in range(self.n_clusters):
#                 # Calculate the distance between each data point and the center of the cluster
#                 distances[:, i] = np.linalg.norm(X - self.centers[i], axis=1)   # Euclidean distance    
            
#             # Avoid division by zero
#             distances = np.fmax(distances, np.finfo(float).eps)
            
#             # Calculate the membership values
#             exp = 2./(self.m-1)
#             # Inverse distance relation for membership calculation
#             tmp = (distances ** (-exp))
            
#             # Normalize the membership values
#             self.u = tmp / np.sum(tmp, axis=1)[:, np.newaxis]

#             # Check convergence
#             if old_centers is not None:
#                 # Check if the centers have converged
#                 if np.all(np.abs(old_centers - self.centers) < self.tol):
#                     break

# class T1FRBS:
#     """
#     Type-1 Fuzzy Rule-Based System (zero-order TSK) for multidimensional input.
    
#     Attributes:
#         n_rules (int): Number of fuzzy rules.
#         n_features (int): Number of input features.
#         centers (np.array): Array of rule centers of shape (n_rules, n_features).
#         sigma (np.array): Standard deviations for each input dimension.
#         p (np.array): Zero-order TSK consequent parameters (constants) for each rule.
#     """
#     def __init__(self, n_rules, n_features, centers=None):
#         self.n_rules = n_rules
#         self.n_features = n_features
#         self.centers = centers
#         self.sigma = None  # array of shape (n_features,)
#         self.p = np.zeros(n_rules)

#     def compute_membership(self, x):
#         """
#         Compute the firing strength for each rule using product t-norm
#         Optimized using NumPy vectorized operations
#         """
#         # Reshape x to 2D array for vectorized operations
#         x = np.array(x).reshape(1, -1)
        
#         # Calculate Gaussian membership functions for all features and rules vectorially
#         diff = x[:, np.newaxis, :] - self.centers
#         exp_term = -0.5 * (diff / self.sigma) ** 2
#         memberships = np.exp(exp_term)
        
#         # Apply product t-norm using prod along last axis
#         mu = np.prod(memberships, axis=2).flatten()
        
#         return mu

#     def predict(self, x):
#         """
#         Predict output using weighted average defuzzification.
#         """ 
#         # Compute the firing strength for each rule
#         mu = self.compute_membership(x)
        
#         # Check if the sum of the membership values is zero
#         if np.sum(mu) == 0:
#             return 0
        
#         # Compute the output using the weighted average defuzzification
#         return np.sum(mu * self.p) / np.sum(mu)

#     def fit(self, x_train, y_train):
#         """
#         Fit the zero-order TSK consequents using the training data.
#         """
#         N, d = x_train.shape
        
#         # If the centers are not provided, use FCMeans to cluster the data
#         if self.centers is None:
#             fcm = FCMeans(n_clusters=self.n_rules)
#             fcm.fit(x_train)
#             self.centers = fcm.centers
        
#         # Compute sigma for each dimension
#         self.sigma = np.zeros(self.n_features)
#         for j in range(self.n_features):
#             # Compute the standard deviation for each dimension
#             x_min = np.min(x_train[:, j])
#             x_max = np.max(x_train[:, j])
#             self.sigma[j] = (x_max - x_min) / (2 * self.n_rules)
        
#         # Learn the consequent parameters
#         self.p = np.zeros(self.n_rules)
#         for i in range(self.n_rules):
#             # Compute the weights for each rule
#             weights = []
#             for x in x_train:
#                 weights.append(self.compute_membership(x)[i])
#             weights = np.array(weights)
            
#             # Check if the sum of the weights is greater than zero
#             if np.sum(weights) > 0:
#                 # Compute the consequent parameter
#                 self.p[i] = np.sum(weights * y_train) / np.sum(weights)
#             else:
#                 # If the sum of the weights is zero, set the consequent parameter to zero
#                 self.p[i] = 0

#     def predict_all(self, x_array):
#         """
#         Predict outputs for an array of input vectors.
#         """
#         # Predict the output for each input vector
#         return np.array([self.predict(x) for x in x_array])

#     def plot_rules(self, name, dim=0, num_points=100):
#         """
#         Plot the Gaussian membership functions for each rule.
#         """
#         if self.sigma is None:
#             raise Exception("Model not fitted yet")
        
#         c_vals = self.centers[:, dim]
#         x_min_dim = np.min(c_vals) - 1
#         x_max_dim = np.max(c_vals) + 1
#         x_range = np.linspace(x_min_dim, x_max_dim, num_points)
        
#         n_cols = math.ceil(math.sqrt(self.n_rules))
#         n_rows = math.ceil(self.n_rules / n_cols)
        
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
#         if self.n_rules == 1:
#             axes = [axes]
#         else:
#             axes = axes.flatten()
        
#         for i in range(self.n_rules):
#             ax = axes[i]
#             center = self.centers[i, dim]
#             mu_vals = gaussian_mf(x_range, center, self.sigma[dim])
            
#             ax.plot(x_range, mu_vals, '-', label='Membership')
#             ax.set_title(f'Rule {i+1}')
#             ax.grid(True, linestyle='--', alpha=0.5)
#             ax.legend()
        
#         for j in range(self.n_rules, len(axes)):
#             axes[j].axis('off')
        
#         plt.tight_layout()
#         plt.savefig(f"{name}-rules.png")

# def gaussian_mf(x, c, sigma):
#     """
#     Gaussian membership function.
#     """
#     return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# def rmse(y_true, y_pred):
#     """
#     Compute the Root Mean Squared Error.
#     """
#     return np.sqrt(np.mean((y_true - y_pred) ** 2))

# def plot_predictions(y_true, y_pred, title):
#     """
#     Plot predicted versus true output values.
#     """
#     plt.figure(figsize=(6, 6))
#     plt.scatter(y_true, y_pred, color='blue', marker='o')
#     min_val = min(np.min(y_true), np.min(y_pred))
#     max_val = max(np.max(y_true), np.max(y_pred))
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
#     plt.xlabel('True Values')
#     plt.ylabel('Predicted Values')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.show()

# def main():
    
#     # Load the dataset from the Excel file
#     df = pd.read_excel("concrete+slump+data.xlsx")


#     # Pre-processing Steps

#     # 1. Handle missing values (if any)
#     df.dropna(inplace=True)         # Drop rows with missing values
#     df.reset_index(drop=True, inplace=True)  # Reset the index after dropping rows

#     # 2. Remove duplicate rows (if present)
#     df.drop_duplicates(inplace=True)

#     # 3. Optionally, scale the input features using StandardScaler (or another scaler)
#     input_features = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]
#     X = df[input_features].values
#     X_normalize = normalizer(X)

#     # 4. Extract output variables (targets)
#     y_slump = df["SLUMP(cm)"].values
#     y_flow  = df["FLOW(cm)"].values
#     y_cs    = df["Compressive Strength (28-day)(Mpa)"].values 

#     n_features = X.shape[1]
#     n_rules = 10
#     n_runs = 50   # Number of random splits to evaluate

#     # Lists to collect RMSE results for each output over multiple runs.
#     rmse_slump_list = []
#     rmse_flow_list = []
#     rmse_cs_list = []

#     for run in range(n_runs):

#         # 2. Randomly split the dataset (80% train, 20% test)
#         indices = np.random.permutation(len(X_normalize))
#         n_train = int(0.8 * len(X_normalize))
#         train_idx = indices[:n_train]
#         test_idx = indices[n_train:]
#         X_train = X_normalize[train_idx]
#         X_test = X_normalize[test_idx]
        
#         # 3. Train one FRBS per output variable.
#         # Slump model
#         frbs_slump = T1FRBS(n_rules, n_features)
#         frbs_slump.fit(X_train, y_slump[train_idx])
#         y_pred_slump = frbs_slump.predict_all(X_test)
#         rmse_slump = rmse(y_slump[test_idx], y_pred_slump)
#         rmse_slump_list.append(rmse_slump)
        
#         # Flow model
#         frbs_flow = T1FRBS(n_rules, n_features)
#         frbs_flow.fit(X_train, y_flow[train_idx])
#         y_pred_flow = frbs_flow.predict_all(X_test)
#         rmse_flow_val = rmse(y_flow[test_idx], y_pred_flow)
#         rmse_flow_list.append(rmse_flow_val)
        
#         # 28-day Compressive Strength model
#         frbs_cs = T1FRBS(n_rules, n_features)
#         frbs_cs.fit(X_train, y_cs[train_idx])
#         y_pred_cs = frbs_cs.predict_all(X_test)
#         rmse_cs_val = rmse(y_cs[test_idx], y_pred_cs)
#         rmse_cs_list.append(rmse_cs_val)
        
    
#     # 4. Report the best (lowest) and mean RMSE for each output variable.
#     print("Results over {} runs:".format(n_runs))
#     print("Slump: Mean RMSE = {:.4f}, Best RMSE = {:.4f}".format(np.mean(rmse_slump_list), np.min(rmse_slump_list)))
#     print("Flow: Mean RMSE = {:.4f}, Best RMSE = {:.4f}".format(np.mean(rmse_flow_list), np.min(rmse_flow_list)))
#     print("28-day Compressive Strength: Mean RMSE = {:.4f}, Best RMSE = {:.4f}".format(np.mean(rmse_cs_list), np.min(rmse_cs_list)))
    
#     # 5. Optionally, plot predictions from one run (for example, the last run)
#     plot_predictions(y_slump[test_idx], y_pred_slump, "Slump: True vs Predicted")
#     plot_predictions(y_flow[test_idx], y_pred_flow, "Flow: True vs Predicted")
#     plot_predictions(y_cs[test_idx], y_pred_cs, "28-day Compressive Strength: True vs Predicted")

#     # 6. Optionally, plot membership functions for a selected input dimension (e.g., dimension 0)
#     frbs_slump.plot_rules("Slump")
#     frbs_cs.plot_rules("Compressive Strength")
#     frbs_flow.plot_rules("Flow")

# if __name__ == "__main__":
#     main()

