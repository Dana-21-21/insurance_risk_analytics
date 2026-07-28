import pandas as pd

from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

#claim frequency

def calculate_claim_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary claim indicator.
    """

    data = df.copy()

    data["ClaimOccurred"] = (data["TotalClaims"] > 0).astype(int)

    return data

#Claim Severity

def calculate_claim_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only policies with claims.
    """

    return df[df["TotalClaims"] > 0].copy()

#Margin

def calculate_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate policy margin.
    """

    data = df.copy()

    data["Margin"] = (
        data["TotalPremium"] -
        data["TotalClaims"]
    )

    return data



#Chi-square Test


def chi_square_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str
):
    """
    Perform a chi-square test.
    """

    table = pd.crosstab(
        df[group_col],
        df[outcome_col]
    )

    chi2, p, dof, expected = chi2_contingency(table)

    return chi2, p



#Independent t-test


def independent_t_test(
    group_a,
    group_b
):
    """
    Independent two-sample t-test.
    """

    statistic, p = ttest_ind(
        group_a,
        group_b,
        equal_var=False,
        nan_policy="omit"
    )

    return statistic, p



#Results Table

def create_results_table(results):
    """
    Convert hypothesis results to a dataframe.
    """

    return pd.DataFrame(results)




def run_hypothesis_test(
    df: pd.DataFrame,
    test_name: str,
    test_type: str,
    group_column: str,
    group_a,
    group_b,
    outcome_column: str,
):
    """
    Run a hypothesis test and return the result as a dictionary.
    """

    data = df[df[group_column].isin([group_a, group_b])].copy()

    if test_type == "chi-square":

        _, p_value = chi_square_test(
            data,
            group_column,
            outcome_column,
        )

        test_used = "Chi-Square"

    elif test_type == "t-test":

        sample_a = data.loc[
            data[group_column] == group_a,
            outcome_column,
        ]

        sample_b = data.loc[
            data[group_column] == group_b,
            outcome_column,
        ]

        _, p_value = independent_t_test(
            sample_a,
            sample_b,
        )

        test_used = "Independent T-Test"

    else:
        raise ValueError("Unsupported test type.")

    decision = (
        "Reject H₀"
        if p_value < 0.05
        else "Fail to Reject H₀"
    )

    return {
        "Hypothesis": test_name,
        "Test": test_used,
        "P-Value": p_value,
        "Decision": decision,
    }


#Claim Frequency Comparison
import matplotlib.pyplot as plt
import seaborn as sns
def plot_claim_frequency(
    df: pd.DataFrame,
    group_column: str,
    figsize=(8, 5),
):
    """
    Plot claim frequency by category.
    """

    frequency = (
        df.groupby(group_column)["ClaimOccurred"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=figsize)

    sns.barplot(
        x=frequency.index,
        y=frequency.values,
    )

    plt.title(f"Claim Frequency by {group_column}")

    plt.xlabel(group_column)

    plt.ylabel("Claim Frequency")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()
    
    
    

    
#Margin Comparison
def plot_margin(
    df: pd.DataFrame,
    group_column: str,
    figsize=(8, 5),
):
    """
    Plot average margin by category.
    """

    margin = (
        df.groupby(group_column)["Margin"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=figsize)

    sns.barplot(
        x=margin.index,
        y=margin.values,
    )

    plt.title(f"Average Margin by {group_column}")

    plt.xlabel(group_column)

    plt.ylabel("Average Margin")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()










