import numpy as np
import matplotlib.pyplot as plt
import math

def data(xtrain, ytrain, xtest, ytest):
    # Print basic information about the training data
    print("Training Data:")
    print(f"  Number of samples: {len(xtrain)}")
    print(f"  x: min = {np.min(xtrain):.3f}, max = {np.max(xtrain):.3f}, mean = {np.mean(xtrain):.3f}, std = {np.std(xtrain):.3f}")
    print(f"  y: min = {np.min(ytrain):.3f}, max = {np.max(ytrain):.3f}, mean = {np.mean(ytrain):.3f}, std = {np.std(ytrain):.3f}\n")
    
    # Print basic information about the testing data
    print("Testing Data:")
    print(f"  Number of samples: {len(xtest)}")
    print(f"  x: min = {np.min(xtest):.3f}, max = {np.max(xtest):.3f}, mean = {np.mean(xtest):.3f}, std = {np.std(xtest):.3f}")
    print(f"  y: min = {np.min(ytest):.3f}, max = {np.max(ytest):.3f}, mean = {np.mean(ytest):.3f}, std = {np.std(ytest):.3f}\n")
    

def triangle_mf(x, a, b, c):
    """
    Triangle membership function.
    Parameters:
        x: Input value
        a: Left point
        b: Center point (peak)
        c: Right point
    """
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)

class T1FRBS:
    """
    Type-1 Fuzzy Rule-Based System (zero-order TSK) with triangle membership functions.

    Attributes:
        n_rules (int): Number of fuzzy rules.
        centers (np.array): Centers of the membership functions (one per rule).
        width (float): Width for the triangle membership functions.
        p (np.array): Zero-order TSK consequent parameters (constants) for each rule.
        xmin (float): Lower bound of the input domain.
        xmax (float): Upper bound of the input domain.
    """
    def __init__(self, n_rules, centers=None, xmin=-2, xmax=2):
        self.n_rules = n_rules
        self.xmin = xmin
        self.xmax = xmax
        # Set the centers of the membership functions
        if centers is None:
            # If centers are not provided, create evenly spaced centers across the input domain
            self.centers = np.linspace(xmin, xmax, n_rules)
        else:
            # If centers are provided, use them as is
            self.centers = centers
            
        # Calculate width for triangles based on domain and number of rules
        span = abs(self.xmax - self.xmin)
        self.width = span / (n_rules - 1)
        self.p = np.zeros(n_rules)  # consequents to be learned

    def compute_membership(self, x):
        """
        For a given input x, compute the membership degrees for each rule.
        """
        # Initialize membership values for each rule
        mu = np.zeros(self.n_rules)
        
        # Compute membership for each rule
        for i, c in enumerate(self.centers):
            # Define triangle points: left(a), center(b), right(c)
            a = c - self.width
            b = c
            c = c + self.width
            
            # Compute membership degree for the current rule
            mu[i] = triangle_mf(x, a, b, c)
        
        return mu

    def predict(self, x):
        """
        Predict for a single x using Type-1 fuzzy logic.
        """
        # Compute membership degrees for the input x
        mu = self.compute_membership(x)
        
        # Perform weighted average defuzzification
        if np.sum(mu) > 0:
            return np.sum(mu * self.p) / np.sum(mu)
        
        # Return 0 if no membership degree is positive
        return 0.0

    def fit(self, x_train, y_train):
        """
        Fit the zero-order TSK consequents.
        """
        # Initialize consequents
        p = np.zeros(self.n_rules)
        
        # Fit each rule
        for i, c in enumerate(self.centers):
            # Calculate triangle points
            a = c - self.width
            b = c
            c = c + self.width
            
            # Compute membership weights for the current rule
            w = np.array([triangle_mf(x, a, b, c) for x in x_train])
            
            # Update consequent if there is any membership
            if np.sum(w) > 0:
                p[i] = np.sum(w * y_train) / np.sum(w)
            else:
                # If no membership, set consequent to 0
                p[i] = 0
        self.p = p

    def predict_all(self, x_array):
        """
        Predict outputs for an array of inputs.
        """
        return np.array([self.predict(x) for x in x_array])

    def plot_rules(self, x_range=np.arange(-2, 2.05, 0.05)):
        """
        Plot each rule's membership function in separate subplots.
        """
        n_rules = self.n_rules
        n_cols = math.ceil(math.sqrt(n_rules))
        n_rows = math.ceil(n_rules / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.array(axes).flatten()

        for i, center in enumerate(self.centers):
            # Calculate triangle points
            a = center - self.width
            b = center
            c = center + self.width
            mu_vals = [triangle_mf(x, a, b, c) for x in x_range]
            
            ax = axes[i]
            ax.plot(x_range, mu_vals, '-', label='Membership')
            ax.set_title(f'Rule {i+1} (center={center:.2f})')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig("rules")
        plt.close()

def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def plot_function_approximation(x_train, y_train, x_test, y_test, x_range, y_pred):
    """
    Plot the function approximation along with the training and test data.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_pred, label='FRBS Approximation', color='green', linewidth=2)
    plt.scatter(x_train, y_train, label='Train Data', color='blue', marker='o', s=50)
    plt.scatter(x_test, y_test, label='Test Data', color='red', marker='x', s=70)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Approximation with IT2 FRBS')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("function_approximation")

def plot_function(frbs, x_range=None):
    """
    Plot the fuzzy system's output over the given range.
    
    Parameters:
        frbs: The fuzzy rule-based system
        x_range: Input range for plotting. If None, uses default range
    """
    if x_range is None:
        # If no range is provided, use the default range
        x_range = np.arange(-2, 2.05, 0.05)
    
    # Calculate predictions
    y_pred = frbs.predict_all(x_range)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_pred, 'b-', linewidth=2, label='Fuzzy System Output')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Type-1 Fuzzy System Response')
    plt.legend()
    plt.tight_layout()
    plt.savefig("function_plot")
    plt.close()

def main():
    # Provided training and test data
    xtrain = np.array([-2, -1.928617603731178, -1.8388106323716946, -1.6505360588614146, -1.6466737987854083, 
                       -1.5257057561784362, -1.505126916572999, -1.2955420401570672, -1.2122777008048495, 
                       -1.0858122491745035, -0.9800400137767338, -0.9779517282635433, -0.9331787789830388, 
                       -0.7874813039993671, -0.619838545990298, -0.4886226045199771, -0.382651422395774, 
                       -0.33300018110610896, -0.08664409599647804, 0.06743694482422713, 0.35682294786440316, 
                       0.39378743445292974, 0.5851822041845041, 0.6804321390567245, 0.7161701249297399, 
                       0.8723862165263401, 0.893034849006435, 0.9333918116453996, 0.9400309986851507, 
                       0.9910626847722677, 1.2247286373398238, 1.2658191715020592, 1.3030635701944662, 
                       1.358983543088455, 1.3593742798418726, 1.3935574671917363, 1.4062142116954033, 
                       1.5388539859334647, 1.6971277518635222, 1.7861198603661284])
    ytrain = np.array([0.9999999999999992, 0.31487794940850466, -0.29666651103951447, 0.08665913205587092, 
                       0.10736120053455869, 0.5176377997731151, 0.5071452098300027, -0.7893502291513371, 
                       -1.1844912264703529, -0.776306607295187, 0.18406664352546748, 0.20271657641210541, 
                       0.5565148316701979, 0.6320946748233786, -0.9758134393879618, -1.4751116374799857, 
                       -0.6382067658058621, 0.004499901071692179, 1.6381326660484838, 0.35096099396214053, 
                       0.12601644334980847, 0.3104637030728553, 0.13238666461451254, -0.6095196293825365, 
                       -0.8693506614049537, -1.0456788261509466, -0.926602149118585, -0.6195847460262386, 
                       -0.5618087599760331, -0.08472370267127548, 0.5525531397237378, 0.2090593933813838, 
                       -0.17218692090734544, -0.7699779902525705, -0.7739470584880559, -1.09399367705507, 
                       -1.1950009789254477, -1.3579923859092413, 0.4103275243772142, 1.4064357239339786])
    
    xtest = np.array([-1.8652295067688538, -1.532894160073611, -1.1179362405895072, -1.1096094984898555, 
                      -0.76632706968909, -0.13255097331976584, 0.3046084462091203, 0.44498061136975187, 
                      0.5660072373556777, 0.7206199322245301, 0.9000763586957743, 1.3457760241567094, 
                      1.4623462150245938, 1.5243013202317912, 1.8850451348731059])
    ytest = np.array([-0.16810844971419459, 0.5145467362049033, -0.9935548408929169, -0.9434730918975127, 
                      0.4871193936655061, 1.7424111202167762, -0.14763621518213493, 0.48411576702695025, 
                      0.25210056265709857, -0.8981040194704278, -0.8795096331494634, -0.6329426445881473, 
                      -1.4827692234839247, -1.429933468059694, 1.72646286123605])
    
    data(xtrain, ytrain, xtest, ytest)

    # Create the T1 FRBS object 
    frbs = T1FRBS(n_rules=24)
    
    # Fit the model using the training data
    frbs.fit(xtrain, ytrain)
    
    # Predict on test data and calculate RMSE
    y_pred_test = frbs.predict_all(xtest)
    test_error = rmse(ytest, y_pred_test)
    print(f"Test RMSE: {test_error:.4f}")
    
    # Plot the approximated function over [-2,2] (step = 0.05)
    x_range = np.arange(-2, 2.05, 0.05)
    y_pred_range = frbs.predict_all(x_range)
    plot_function_approximation(xtrain, ytrain, xtest, ytest, x_range, y_pred_range)
    
    # Plot the membership functions for each rule
    frbs.plot_rules()
    
    # Plot the function
    plot_function(frbs)

if __name__ == "__main__":
    main()
    