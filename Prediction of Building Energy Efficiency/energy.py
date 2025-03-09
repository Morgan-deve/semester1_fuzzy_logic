import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def normalizer(df: np.ndarray):
    """Normalize data between 0 and 1"""
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

class FCMeans:
    """Implementation of Fuzzy C-means clustering algorithm"""
    def __init__(self, n_clusters=3, max_iter=100, m=2, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m  # fuzziness parameter
        self.tol = tol
        self.random_state = random_state
        self.centers = None
        self.u = None  # membership matrix

    def fit(self, X):
        """Fit the fuzzy c-means model"""
        if self.random_state is not None:
            # Set the random seed for reproducibility
            np.random.seed(self.random_state)
        
        N = X.shape[0]
        
        # Initialize membership matrix
        self.u = np.random.rand(N, self.n_clusters)
        # Normalize so that sum of memberships for each data point equals 1
        self.u = self.u / np.sum(self.u, axis=1)[:, np.newaxis]

        for _ in range(self.max_iter):
            # Update membership matrix
            u_m = self.u ** self.m
            
            # Update centers
            old_centers = self.centers if self.centers is not None else None
            # Compute new cluster centers using weighted average of data points
            self.centers = (u_m.T @ X) / np.sum(u_m, axis=0)[:, np.newaxis]
            
            # Update membership matrix
            distances = np.zeros((N, self.n_clusters))
            for i in range(self.n_clusters):
                # Calculate distances between data points and centers
                distances[:, i] = np.linalg.norm(X - self.centers[i], axis=1)
            
            # Avoid division by zero
            distances = np.fmax(distances, np.finfo(float).eps)
            
            # Update membership matrix
            exp = 2./(self.m-1)
            
            # Update membership matrix
            tmp = (distances ** (-exp))
            self.u = tmp / np.sum(tmp, axis=1)[:, np.newaxis]

            if old_centers is not None:
                # Check if the centers have converged
                if np.all(np.abs(old_centers - self.centers) < self.tol):
                    break

class T1FRBS:
    """
    Type-1 Fuzzy Rule-Based System (zero-order TSK) for multidimensional input
    """
    def __init__(self, n_rules, n_features, centers=None):
        self.n_rules = n_rules
        self.n_features = n_features
        self.centers = centers
        self.sigma = None
        self.p = np.zeros(n_rules)

    def compute_membership(self, x):
        """Compute firing strength for each rule"""
        # Reshape input to match the expected shape 
        x = np.array(x).reshape(1, -1)
        
        # Calculate differences between input and centers
        diff = x[:, np.newaxis, :] - self.centers
        
        # Compute Gaussian membership values
        exp_term = -0.5 * (diff / self.sigma) ** 2
        memberships = np.exp(exp_term)
        
        # Compute product of memberships
        mu = np.prod(memberships, axis=2).flatten()
        return mu

    def predict(self, x):
        """Predict output using weighted average defuzzification"""
        # Compute membership values for each rule
        mu = self.compute_membership(x)
        
        # Avoid division by zero
        if np.sum(mu) == 0:
            return 0
        
        # Compute weighted average
        return np.sum(mu * self.p) / np.sum(mu)

    def fit(self, x_train, y_train):
        """Train the fuzzy system parameters"""
        # Get the number of samples and features
        N, d = x_train.shape
        
        # If centers are not provided, use Fuzzy C-means to find them
        if self.centers is None:
            fcm = FCMeans(n_clusters=self.n_rules)
            fcm.fit(x_train)
            self.centers = fcm.centers
        
        # Calculate sigma for each dimension
        self.sigma = np.zeros(self.n_features)
        for j in range(self.n_features):
            x_min = np.min(x_train[:, j])
            x_max = np.max(x_train[:, j])
            self.sigma[j] = (x_max - x_min) / (2 * self.n_rules)
        
        # Learn consequent parameters
        self.p = np.zeros(self.n_rules)

        # Iterate over each rule
        for i in range(self.n_rules):
            # Compute weights for each sample
            weights = []
            for x in x_train:
                # Compute membership value for each rule
                weights.append(self.compute_membership(x)[i])
            weights = np.array(weights)

            # Avoid division by zero
            if np.sum(weights) > 0:
                # Compute weighted average
                self.p[i] = np.sum(weights * y_train) / np.sum(weights)
            else:
                self.p[i] = 0

    def predict_all(self, x_array):
        """Predict for an array of inputs"""
        # Predict output for each input in the array
        return np.array([self.predict(x) for x in x_array])

def rmse(y_true, y_pred):
    """Calculate RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def plot_predictions(y_true, y_pred, title):
    """Plot predicted vs actual values"""
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

def plot_fuzzy_sets(frbs, feature_names):
    """Plot Gaussian membership functions for each feature"""
    n_features = frbs.n_features
    n_rules = frbs.n_rules
    
    # Create a figure with subplots for each feature
    fig, axes = plt.subplots(((n_features + 1) // 2), 2, figsize=(12, 2*((n_features + 1) // 2)))
    axes = axes.flatten()
    
    # For each feature
    for j in range(n_features):
        # Generate points for x-axis (normalized space)
        x = np.linspace(0, 1, 200)
        
        # Plot membership functions for each rule
        for i in range(n_rules):
            center = frbs.centers[i, j]
            sigma = frbs.sigma[j]
            
            # Calculate Gaussian membership values
            y = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            
            # Plot the membership function
            axes[j].plot(x, y, label=f'Rule {i+1}')
        
        axes[j].set_title(f'Feature: {feature_names[j]}')
        axes[j].set_xlabel('Normalized Input')
        axes[j].set_ylabel('Membership Degree')
        axes[j].grid(True, linestyle='--', alpha=0.5)
    
    # Remove any extra subplots
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    df = pd.read_excel("ENB2012_data.xlsx")

    # Preprocessing
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Define input features based on actual column names in your Excel file
    # You may need to adjust these names to match exactly with your Excel file
    input_features = [
        "X1",  # Relative Compactness
        "X2",  # Surface Area
        "X3",  # Wall Area
        "X4",  # Roof Area
        "X5",  # Overall Height
        "X6",  # Orientation
        "X7",  # Glazing Area
        "X8"   # Glazing Area Distribution
    ]
    
    # Output variables
    y_heating = df["Y1"].values  # Heating Load
    y_cooling = df["Y2"].values  # Cooling Load

    # Extract input features and normalize them
    X = df[input_features].values
    X_normalize = normalizer(X)

    # Get the number of features and rules
    n_features = X.shape[1]
    n_rules = 10
    
    # Number of runs for averaging results
    n_runs = 50

    # Initialize lists to store RMSE values
    rmse_heating_list = []
    rmse_cooling_list = []

    for run in range(n_runs):
        # Random split
        indices = np.random.permutation(len(X_normalize))
        n_train = int(0.8 * len(X_normalize))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        # Extract training and test data
        X_train = X_normalize[train_idx]
        X_test = X_normalize[test_idx]
        
        # Heating Load model
        frbs_heating = T1FRBS(n_rules, n_features)
        frbs_heating.fit(X_train, y_heating[train_idx])
        y_pred_heating = frbs_heating.predict_all(X_test)
        rmse_heating = rmse(y_heating[test_idx], y_pred_heating)
        rmse_heating_list.append(rmse_heating)
        
        # Cooling Load model
        frbs_cooling = T1FRBS(n_rules, n_features)
        frbs_cooling.fit(X_train, y_cooling[train_idx])
        y_pred_cooling = frbs_cooling.predict_all(X_test)
        rmse_cooling = rmse(y_cooling[test_idx], y_pred_cooling)
        rmse_cooling_list.append(rmse_cooling)

    # Report results
    print(f"Results over {n_runs} runs:")
    print(f"Heating Load: Mean RMSE = {np.mean(rmse_heating_list):.4f}, Best RMSE = {np.min(rmse_heating_list):.4f}")
    print(f"Cooling Load: Mean RMSE = {np.mean(rmse_cooling_list):.4f}, Best RMSE = {np.min(rmse_cooling_list):.4f}")

    # Plot results for the last run
    plot_predictions(y_heating[test_idx], y_pred_heating, "Heating Load: True vs Predicted")
    plot_predictions(y_cooling[test_idx], y_pred_cooling, "Cooling Load: True vs Predicted")

    # After training the models, visualize the fuzzy sets
    print("\nVisualizing Fuzzy Sets for Heating Load Model:")
    plot_fuzzy_sets(frbs_heating, input_features)
    print("\nVisualizing Fuzzy Sets for Cooling Load Model:")
    plot_fuzzy_sets(frbs_cooling, input_features)

if __name__ == "__main__":
    main()
    
    


# import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd


# def normalizer(df: np.ndarray):
#     """Normalize data between 0 and 1"""
#     normalized_df = (df - df.min()) / (df.max() - df.min())
#     return normalized_df

# class FCMeans:
#     """Implementation of Fuzzy C-means clustering algorithm"""
#     def __init__(self, n_clusters=3, max_iter=100, m=2, tol=1e-4, random_state=None):
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.m = m  # fuzziness parameter
#         self.tol = tol
#         self.random_state = random_state
#         self.centers = None
#         self.u = None  # membership matrix

#     def fit(self, X):
#         """Fit the fuzzy c-means model"""
#         if self.random_state is not None:
#             # Set the random seed for reproducibility
#             np.random.seed(self.random_state)
        
#         N = X.shape[0]
        
#         # Initialize membership matrix
#         self.u = np.random.rand(N, self.n_clusters)
#         # Normalize so that sum of memberships for each data point equals 1
#         self.u = self.u / np.sum(self.u, axis=1)[:, np.newaxis]

#         for _ in range(self.max_iter):
#             # Update membership matrix
#             u_m = self.u ** self.m
            
#             # Update centers
#             old_centers = self.centers if self.centers is not None else None
#             # Compute new cluster centers using weighted average of data points
#             self.centers = (u_m.T @ X) / np.sum(u_m, axis=0)[:, np.newaxis]
            
#             # Update membership matrix
#             distances = np.zeros((N, self.n_clusters))
#             for i in range(self.n_clusters):
#                 # Calculate distances between data points and centers
#                 distances[:, i] = np.linalg.norm(X - self.centers[i], axis=1)
            
#             # Avoid division by zero
#             distances = np.fmax(distances, np.finfo(float).eps)
            
#             # Update membership matrix
#             exp = 2./(self.m-1)
            
#             # Update membership matrix
#             tmp = (distances ** (-exp))
#             self.u = tmp / np.sum(tmp, axis=1)[:, np.newaxis]

#             if old_centers is not None:
#                 # Check if the centers have converged
#                 if np.all(np.abs(old_centers - self.centers) < self.tol):
#                     break

# class T1FRBS:
#     """
#     Type-1 Fuzzy Rule-Based System (zero-order TSK) for multidimensional input
#     """
#     def __init__(self, n_rules, n_features, centers=None):
#         self.n_rules = n_rules
#         self.n_features = n_features
#         self.centers = centers
#         self.sigma = None
#         self.p = np.zeros(n_rules)

#     def compute_membership(self, x):
#         """Compute firing strength for each rule"""
#         # Reshape input to match the expected shape 
#         x = np.array(x).reshape(1, -1)
        
#         # Calculate differences between input and centers
#         diff = x[:, np.newaxis, :] - self.centers
        
#         # Compute Gaussian membership values
#         exp_term = -0.5 * (diff / self.sigma) ** 2
#         memberships = np.exp(exp_term)
        
#         # Compute product of memberships
#         mu = np.prod(memberships, axis=2).flatten()
#         return mu

#     def predict(self, x):
#         """Predict output using weighted average defuzzification"""
#         # Compute membership values for each rule
#         mu = self.compute_membership(x)
        
#         # Avoid division by zero
#         if np.sum(mu) == 0:
#             return 0
        
#         # Compute weighted average
#         return np.sum(mu * self.p) / np.sum(mu)

#     def fit(self, x_train, y_train):
#         """Train the fuzzy system parameters"""
#         # Get the number of samples and features
#         N, d = x_train.shape
        
#         # If centers are not provided, use Fuzzy C-means to find them
#         if self.centers is None:
#             fcm = FCMeans(n_clusters=self.n_rules)
#             fcm.fit(x_train)
#             self.centers = fcm.centers
        
#         # Calculate sigma for each dimension
#         self.sigma = np.zeros(self.n_features)
#         for j in range(self.n_features):
#             x_min = np.min(x_train[:, j])
#             x_max = np.max(x_train[:, j])
#             self.sigma[j] = (x_max - x_min) / (2 * self.n_rules)
        
#         # Learn consequent parameters
#         self.p = np.zeros(self.n_rules)

#         # Iterate over each rule
#         for i in range(self.n_rules):
#             # Compute weights for each sample
#             weights = []
#             for x in x_train:
#                 # Compute membership value for each rule
#                 weights.append(self.compute_membership(x)[i])
#             weights = np.array(weights)

#             # Avoid division by zero
#             if np.sum(weights) > 0:
#                 # Compute weighted average
#                 self.p[i] = np.sum(weights * y_train) / np.sum(weights)
#             else:
#                 self.p[i] = 0

#     def predict_all(self, x_array):
#         """Predict for an array of inputs"""
#         # Predict output for each input in the array
#         return np.array([self.predict(x) for x in x_array])

# def rmse(y_true, y_pred):
#     """Calculate RMSE"""
#     return np.sqrt(np.mean((y_true - y_pred) ** 2))

# def plot_predictions(y_true, y_pred, title):
#     """Plot predicted vs actual values"""
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
#     # Load data
#     df = pd.read_excel("ENB2012_data.xlsx")

#     # Preprocessing
#     df.dropna(inplace=True)
#     df.drop_duplicates(inplace=True)

#     # Define input features based on actual column names in your Excel file
#     # You may need to adjust these names to match exactly with your Excel file
#     input_features = [
#         "X1",  # Relative Compactness
#         "X2",  # Surface Area
#         "X3",  # Wall Area
#         "X4",  # Roof Area
#         "X5",  # Overall Height
#         "X6",  # Orientation
#         "X7",  # Glazing Area
#         "X8"   # Glazing Area Distribution
#     ]
    
#     # Output variables
#     y_heating = df["Y1"].values  # Heating Load
#     y_cooling = df["Y2"].values  # Cooling Load

#     # Extract input features and normalize them
#     X = df[input_features].values
#     X_normalize = normalizer(X)

#     # Get the number of features and rules
#     n_features = X.shape[1]
#     n_rules = 10
    
#     # Number of runs for averaging results
#     n_runs = 50

#     # Initialize lists to store RMSE values
#     rmse_heating_list = []
#     rmse_cooling_list = []

#     for run in range(n_runs):
#         # Random split
#         indices = np.random.permutation(len(X_normalize))
#         n_train = int(0.8 * len(X_normalize))
#         train_idx = indices[:n_train]
#         test_idx = indices[n_train:]
        
#         # Extract training and test data
#         X_train = X_normalize[train_idx]
#         X_test = X_normalize[test_idx]
        
#         # Heating Load model
#         frbs_heating = T1FRBS(n_rules, n_features)
#         frbs_heating.fit(X_train, y_heating[train_idx])
#         y_pred_heating = frbs_heating.predict_all(X_test)
#         rmse_heating = rmse(y_heating[test_idx], y_pred_heating)
#         rmse_heating_list.append(rmse_heating)
        
#         # Cooling Load model
#         frbs_cooling = T1FRBS(n_rules, n_features)
#         frbs_cooling.fit(X_train, y_cooling[train_idx])
#         y_pred_cooling = frbs_cooling.predict_all(X_test)
#         rmse_cooling = rmse(y_cooling[test_idx], y_pred_cooling)
#         rmse_cooling_list.append(rmse_cooling)

#     # Report results
#     print(f"Results over {n_runs} runs:")
#     print(f"Heating Load: Mean RMSE = {np.mean(rmse_heating_list):.4f}, Best RMSE = {np.min(rmse_heating_list):.4f}")
#     print(f"Cooling Load: Mean RMSE = {np.mean(rmse_cooling_list):.4f}, Best RMSE = {np.min(rmse_cooling_list):.4f}")

#     # Plot results for the last run
#     plot_predictions(y_heating[test_idx], y_pred_heating, "Heating Load: True vs Predicted")
#     plot_predictions(y_cooling[test_idx], y_pred_cooling, "Cooling Load: True vs Predicted")

# if __name__ == "__main__":
#     main()