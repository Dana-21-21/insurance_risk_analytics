import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summarize_data(df: pd.DataFrame) -> None:
    """
    Display a summary of the dataset.
    """

    print("=" * 70)
    print("DATASET SHAPE")
    print("=" * 70)
    print(df.shape)

    print("\n")

    print("=" * 70)
    print("DATA TYPES")
    print("=" * 70)
    print(df.dtypes)

    print("\n")

    print("=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    display(df.describe(include="all").T)
    
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display missing values and duplicate records.
    """

    missing = (
        pd.DataFrame({
            "Missing Values": df.isnull().sum(),
            "Percentage": (df.isnull().mean() * 100).round(2)
        })
        .sort_values("Missing Values", ascending=False)
    )

    duplicates = df.duplicated().sum()

    print("=" * 70)
    print(f"Duplicate Rows: {duplicates}")
    print("=" * 70)

    return missing
   
def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary statistics for numerical columns.
    """

    return df.describe().T

def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary statistics for categorical columns.
    """

    return df.describe(include="object").T

def plot_numeric_distribution(df: pd.DataFrame, column: str) -> None:
    """
    Plot the distribution of a numerical feature.
    """

    plt.figure(figsize=(10, 6))

    sns.histplot(df[column], bins=30, kde=True)

    plt.title(f"Distribution of {column}")

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()
    
def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10
) -> None:
    """
    Plot the top categories for a categorical feature.
    """

    plt.figure(figsize=(10, 6))

    (
        df[column]
        .value_counts()
        .head(top_n)
        .plot(kind="bar")
    )

    plt.title(f"{column} Distribution")

    plt.xlabel(column)

    plt.ylabel("Count")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()
    
    
    
    
    #Correlation Matrix
    
def plot_correlation_matrix(df: pd.DataFrame, columns: list) -> None:
    """Plot the correlation matrix for selected numerical columns."""
    corr = df[columns].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    
    
    
#Scatter Plot
    
    
def plot_scatter(df: pd.DataFrame, x: str, y: str) -> None:
    """Scatter plot between two numerical variables."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        alpha=0.5
    )
    plt.title(f"{y} vs {x}")
    plt.tight_layout()
    plt.show()
    
    
#Average of a Numerical Variable by Category
def plot_average_by_category(
    df: pd.DataFrame,
    category: str,
    value: str,
    top_n: int = None
) -> None:
    """Plot the average of a numerical variable grouped by a categorical feature."""
    summary = (
        df.groupby(category)[value]
        .mean()
        .sort_values(ascending=False)
    )

    if top_n:
        summary = summary.head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=summary.index,
        y=summary.values
    )
    plt.title(f"Average {value} by {category}")
    plt.xlabel(category)
    plt.ylabel(f"Average {value}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
# Top Categories
def plot_top_categories(
    df: pd.DataFrame,
    category: str,
    top_n: int = 10
) -> None:
    """Plot the most frequent categories."""
    counts = df[category].value_counts().head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=counts.index,
        y=counts.values
    )
    plt.title(f"Top {top_n} {category}")
    plt.xlabel(category)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    

#Outlier Detection    
def plot_boxplot(
    df: pd.DataFrame,
    column: str
) -> None:
    """
    Plot a box plot for a numerical feature.
    """

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        x=df[column]
    )

    plt.title(f"Box Plot of {column}")

    plt.tight_layout()

    plt.show()
    
    

#Calculate Loss Ratio

def calculate_loss_ratio(
    df: pd.DataFrame,
    premium_col: str = "TotalPremium",
    claims_col: str = "TotalClaims"
) -> float:
    """
    Calculate the overall portfolio loss ratio.
    """

    total_premium = df[premium_col].sum()
    total_claims = df[claims_col].sum()

    if total_premium == 0:
        return 0

    return total_claims / total_premium


#Loss Ratio by Group

def loss_ratio_by_group(
    df: pd.DataFrame,
    group_col: str,
    premium_col: str = "TotalPremium",
    claims_col: str = "TotalClaims"
) -> pd.DataFrame:
    """
    Calculate loss ratio grouped by a categorical variable.
    """

    summary = (
        df.groupby(group_col)[[premium_col, claims_col]]
        .sum()
    )

    summary["LossRatio"] = (
        summary[claims_col] /
        summary[premium_col]
    )

    return summary.sort_values(
        "LossRatio",
        ascending=False
    )



#Temporal Trends

def monthly_claim_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly premium and claim trends.
    """

    monthly = (
        df.groupby("TransactionMonth")
        .agg(
            TotalPremium=("TotalPremium", "sum"),
            TotalClaims=("TotalClaims", "sum"),
            ClaimCount=("TotalClaims", lambda x: (x > 0).sum())
        )
        .reset_index()
    )

    return monthly


#Highest and Lowest Claim Vehicles

def vehicle_claim_summary(
    df: pd.DataFrame,
    group_col: str = "make"
) -> pd.DataFrame:
    """
    Average claim amount by vehicle make or model.
    """

    summary = (
        df.groupby(group_col)["TotalClaims"]
        .mean()
        .sort_values(ascending=False)
    )

    return summary









