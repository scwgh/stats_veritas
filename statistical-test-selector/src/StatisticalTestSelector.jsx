import React, { useState } from 'react';
import { ChevronRight, RotateCcw, BookOpen, CheckCircle, AlertCircle, Info } from 'lucide-react';

const StatisticalTestSelector = () => {
  const [currentStep, setCurrentStep] = useState('start');
  const [path, setPath] = useState([]);

  const decisionTree = {
    start: {
      question: "What type of analysis do you want to perform?",
      options: [
        { text: "Compare groups/methods", next: "compare_type", description: "Test differences between groups or measurement methods" },
        { text: "Test data distribution", next: "distribution_type", description: "Check if data follows a specific distribution" },
        { text: "Analyze relationships", next: "relationship_type", description: "Examine correlations or regression relationships" },
        { text: "Quality control", next: "qc_type", description: "Monitor process stability and detect shifts" }
      ]
    },
    compare_type: {
      question: "How many groups are you comparing?",
      options: [
        { text: "2 groups", next: "two_groups", description: "Compare two independent or paired groups" },
        { text: "3+ groups", next: "multiple_groups", description: "Compare three or more groups" },
        { text: "2 methods", next: "method_comparison", description: "Compare measurement methods or instruments" }
      ]
    },
    two_groups: {
      question: "What type of data do you have?",
      options: [
        { text: "Continuous/Numeric", next: "two_groups_continuous", description: "Measurements like height, weight, concentration" },
        { text: "Categorical/Count", next: "two_groups_categorical", description: "Categories, frequencies, or proportions" }
      ]
    },
    two_groups_continuous: {
      question: "Are the groups paired or independent?",
      options: [
        { text: "Paired/Matched", next: "paired_continuous", description: "Same subjects measured twice or matched pairs" },
        { text: "Independent", next: "independent_continuous", description: "Different subjects in each group" }
      ]
    },
    paired_continuous: {
      question: "Is the difference data normally distributed?",
      options: [
        { 
          text: "Yes (Normal)", 
          next: "result", 
          result: { 
            test: "Paired T-Test", 
            page: "t_test_page", 
            description: "Use when paired differences follow normal distribution",
            assumptions: ["Differences are normally distributed", "Data are paired", "Independence of observations"],
            when_to_use: "When comparing two related groups (before/after, matched pairs) with normally distributed differences",
            example: "Comparing blood pressure before and after treatment in the same patients",
            alternatives: ["Wilcoxon Signed-Rank Test (if non-normal)", "Sign Test (if only direction matters)"]
          } 
        },
        { 
          text: "No (Non-normal)", 
          next: "result", 
          result: { 
            test: "Wilcoxon Signed-Rank Test", 
            page: "wilcoxon_page", 
            description: "Non-parametric alternative for paired comparisons",
            assumptions: ["Data are paired", "Differences are symmetric around median", "Independence of observations"],
            when_to_use: "When comparing two related groups with non-normally distributed differences",
            example: "Comparing pain scores before and after treatment when data is skewed",
            alternatives: ["Paired T-Test (if differences are normal)", "Sign Test (if symmetry assumption violated)"]
          } 
        }
      ]
    },
    independent_continuous: {
      question: "Are both groups normally distributed with equal variances?",
      options: [
        { 
          text: "Yes", 
          next: "result", 
          result: { 
            test: "Independent T-Test", 
            page: "t_test_page", 
            description: "Standard test for comparing two independent normal groups",
            assumptions: ["Normal distribution in both groups", "Equal variances", "Independence of observations"],
            when_to_use: "When comparing means of two independent groups with normal distributions and equal variances",
            example: "Comparing heights between males and females",
            alternatives: ["Welch's T-Test (unequal variances)", "Mann-Whitney U Test (non-normal)"]
          } 
        },
        { 
          text: "Normal but unequal variances", 
          next: "result", 
          result: { 
            test: "Welch's T-Test", 
            page: "welch_t_test_page", 
            description: "T-test variant for unequal variances",
            assumptions: ["Normal distribution in both groups", "Unequal variances allowed", "Independence of observations"],
            when_to_use: "When comparing means of two independent normal groups with different variances",
            example: "Comparing test scores between two schools with different variability",
            alternatives: ["Independent T-Test (if variances are equal)", "Mann-Whitney U Test (non-normal)"]
          } 
        },
        { 
          text: "Non-normal distribution", 
          next: "result", 
          result: { 
            test: "Mann-Whitney U Test", 
            page: "mann_whitney_page", 
            description: "Non-parametric test for independent groups",
            assumptions: ["Independence of observations", "Ordinal or continuous data", "Same shape distributions (for median comparison)"],
            when_to_use: "When comparing two independent groups with non-normal distributions",
            example: "Comparing customer satisfaction ratings between two products",
            alternatives: ["Independent T-Test (if normal)", "Permutation Test", "Bootstrap Test"]
          } 
        }
      ]
    },
    two_groups_categorical: {
      question: "What type of categorical data?",
      options: [
        { 
          text: "2x2 contingency table", 
          next: "chi_square_type", 
          description: "Two categorical variables with 2 categories each" 
        },
        { 
          text: "Proportions", 
          next: "proportion_test", 
          description: "Compare proportions between two groups" 
        },
        { 
          text: "Larger contingency table", 
          next: "result", 
          result: { 
            test: "Chi-Squared Test of Independence", 
            page: "chi_squared_page", 
            description: "Test association between two categorical variables",
            assumptions: ["Expected frequencies ≥ 5 in most cells", "Independence of observations", "Random sampling"],
            when_to_use: "When testing association between two categorical variables",
            example: "Testing if treatment outcome is associated with patient gender",
            alternatives: ["Fisher's Exact Test (small samples)", "Monte Carlo Chi-Square"]
          } 
        }
      ]
    },
    chi_square_type: {
      question: "What are your sample characteristics?",
      options: [
        { 
          text: "Large sample (all expected frequencies ≥ 5)", 
          next: "result", 
          result: { 
            test: "Chi-Squared Test", 
            page: "chi_squared_page", 
            description: "Test association between two categorical variables",
            assumptions: ["Expected frequencies ≥ 5 in all cells", "Independence of observations", "Random sampling"],
            when_to_use: "When testing association in 2x2 tables with adequate sample size",
            example: "Testing if smoking is associated with lung disease",
            alternatives: ["Fisher's Exact Test (small samples)", "Yates' Continuity Correction"]
          } 
        },
        { 
          text: "Small sample or low expected frequencies", 
          next: "result", 
          result: { 
            test: "Fisher's Exact Test", 
            page: "fisher_exact_page", 
            description: "Exact test for 2x2 contingency tables",
            assumptions: ["Independence of observations", "Fixed marginal totals", "Random sampling"],
            when_to_use: "When sample size is small or expected frequencies < 5 in 2x2 tables",
            example: "Testing association between rare disease and exposure with small sample",
            alternatives: ["Chi-Squared Test (larger samples)", "Mid-P Exact Test"]
          } 
        }
      ]
    },
    proportion_test: {
      question: "How many proportions are you comparing?",
      options: [
        { 
          text: "One proportion to a known value", 
          next: "result", 
          result: { 
            test: "One-Sample Z-Test for Proportion", 
            page: "z_test_proportion_page", 
            description: "Test if a sample proportion differs from a known value",
            assumptions: ["Large sample size (np ≥ 5 and n(1-p) ≥ 5)", "Random sampling", "Independence"],
            when_to_use: "When testing if a sample proportion equals a hypothesized value",
            example: "Testing if 50% of customers prefer a new product design",
            alternatives: ["Binomial Exact Test (small samples)", "Wilson Score Interval"]
          } 
        },
        { 
          text: "Two proportions", 
          next: "result", 
          result: { 
            test: "Two-Sample Z-Test for Proportions", 
            page: "z_test_proportions_page", 
            description: "Compare proportions between two groups",
            assumptions: ["Large sample sizes", "Independence between and within groups", "Random sampling"],
            when_to_use: "When comparing success rates between two independent groups",
            example: "Comparing cure rates between two treatments",
            alternatives: ["Fisher's Exact Test (small samples)", "Chi-Squared Test"]
          } 
        }
      ]
    },
    multiple_groups: {
      question: "What type of data and design?",
      options: [
        { 
          text: "Continuous, 1 factor", 
          next: "anova_assumptions", 
          description: "Compare means across multiple groups with one factor"
        },
        { 
          text: "Continuous, 2+ factors", 
          next: "anova_design", 
          description: "Multiple factors affecting the outcome" 
        },
        { 
          text: "Non-normal continuous", 
          next: "result", 
          result: { 
            test: "Kruskal-Wallis Test", 
            page: "kruskal_page", 
            description: "Non-parametric alternative to one-way ANOVA",
            assumptions: ["Independence of observations", "Same shape distributions", "Ordinal or continuous data"],
            when_to_use: "When comparing multiple groups with non-normal data",
            example: "Comparing customer satisfaction across multiple stores",
            alternatives: ["One-Way ANOVA (if normal)", "Mood's Median Test", "Permutation ANOVA"]
          } 
        },
        { 
          text: "Categorical data", 
          next: "result", 
          result: { 
            test: "Chi-Squared Test of Independence", 
            page: "chi_squared_page", 
            description: "Test association in contingency tables",
            assumptions: ["Expected frequencies ≥ 5 in most cells", "Independence of observations", "Random sampling"],
            when_to_use: "When testing association between categorical variables with multiple categories",
            example: "Testing if job satisfaction varies by department",
            alternatives: ["Fisher's Exact Test (small samples)", "Monte Carlo Chi-Square"]
          } 
        }
      ]
    },
    anova_assumptions: {
      question: "Do your data meet ANOVA assumptions?",
      options: [
        { 
          text: "Yes (Normal, equal variances)", 
          next: "result", 
          result: { 
            test: "One-Way ANOVA", 
            page: "one_way_anova_page", 
            description: "Compare means across multiple groups",
            assumptions: ["Normal distribution in all groups", "Equal variances (homoscedasticity)", "Independence of observations"],
            when_to_use: "When comparing means of 3+ independent groups with normal data",
            example: "Comparing average test scores across multiple teaching methods",
            alternatives: ["Kruskal-Wallis Test (non-normal)", "Welch's ANOVA (unequal variances)"]
          } 
        },
        { 
          text: "Normal but unequal variances", 
          next: "result", 
          result: { 
            test: "Welch's ANOVA", 
            page: "welch_anova_page", 
            description: "ANOVA for groups with unequal variances",
            assumptions: ["Normal distribution in all groups", "Unequal variances allowed", "Independence of observations"],
            when_to_use: "When comparing means of 3+ groups with normal but heteroscedastic data",
            example: "Comparing income across regions where variability differs",
            alternatives: ["One-Way ANOVA (if equal variances)", "Kruskal-Wallis Test (non-normal)"]
          } 
        },
        { 
          text: "Violations of assumptions", 
          next: "result", 
          result: { 
            test: "Kruskal-Wallis Test", 
            page: "kruskal_page", 
            description: "Non-parametric alternative to one-way ANOVA",
            assumptions: ["Independence of observations", "Same shape distributions", "Ordinal or continuous data"],
            when_to_use: "When ANOVA assumptions are violated",
            example: "Comparing median response times across multiple conditions",
            alternatives: ["Robust ANOVA", "Permutation ANOVA", "Transformation + ANOVA"]
          } 
        }
      ]
    },
    anova_design: {
      question: "What type of factorial design?",
      options: [
        { 
          text: "Two crossed factors", 
          next: "result", 
          result: { 
            test: "Two-Way ANOVA", 
            page: "two_way_anova_page", 
            description: "Analyze main effects and interactions of two factors",
            assumptions: ["Normal distribution", "Equal variances", "Independence", "No extreme outliers"],
            when_to_use: "When examining effects of two independent factors and their interaction",
            example: "Analyzing effects of both gender and treatment on blood pressure",
            alternatives: ["Mixed-Effects Models", "MANOVA (multiple outcomes)"]
          } 
        },
        { 
          text: "Repeated measures", 
          next: "result", 
          result: { 
            test: "Repeated Measures ANOVA", 
            page: "repeated_measures_anova_page", 
            description: "ANOVA for within-subjects designs",
            assumptions: ["Sphericity", "Normal distribution", "No missing data patterns"],
            when_to_use: "When same subjects are measured multiple times",
            example: "Analyzing changes in performance across multiple time points",
            alternatives: ["Mixed-Effects Models", "MANOVA", "Friedman Test (non-parametric)"]
          } 
        },
        { 
          text: "Multiple factors (3+)", 
          next: "result", 
          result: { 
            test: "Multi-Way ANOVA", 
            page: "multi_way_anova_page", 
            description: "ANOVA with three or more factors",
            assumptions: ["Normal distribution", "Equal variances", "Independence", "Adequate sample size"],
            when_to_use: "When examining effects of multiple factors simultaneously",
            example: "Analyzing effects of treatment, age group, and gender on outcome",
            alternatives: ["Mixed-Effects Models", "Simplified factorial designs"]
          } 
        }
      ]
    },
    method_comparison: {
      question: "What aspect of method comparison?",
      options: [
        { 
          text: "Agreement analysis", 
          next: "result", 
          result: { 
            test: "Bland-Altman Analysis", 
            page: "bland_altman_page", 
            description: "Assess agreement between two measurement methods",
            assumptions: ["Differences are normally distributed", "Constant variance across measurement range"],
            when_to_use: "When assessing if two methods can be used interchangeably",
            example: "Comparing two blood pressure measurement devices",
            alternatives: ["Intraclass Correlation", "Concordance Correlation Coefficient"]
          } 
        },
        { 
          text: "Regression analysis", 
          next: "regression_type", 
          description: "Examine relationship between methods" 
        },
        { 
          text: "Variance comparison", 
          next: "variance_comparison", 
          description: "Compare variability between methods" 
        }
      ]
    },
    regression_type: {
      question: "What type of regression is appropriate?",
      options: [
        { 
          text: "Both methods have measurement error", 
          next: "result", 
          result: { 
            test: "Deming Regression", 
            page: "deming_page", 
            description: "Regression when both X and Y have measurement error",
            assumptions: ["Linear relationship", "Constant error ratio", "Normal errors"],
            when_to_use: "When both measurement methods have known or equal measurement errors",
            example: "Comparing two laboratory instruments with known precision",
            alternatives: ["Passing-Bablok Regression", "Weighted Deming Regression"]
          } 
        },
        { 
          text: "Non-parametric regression", 
          next: "result", 
          result: { 
            test: "Passing-Bablok Regression", 
            page: "passing_bablok_page", 
            description: "Robust non-parametric regression method",
            assumptions: ["No specific distribution requirements", "Linear relationship"],
            when_to_use: "When data doesn't meet parametric assumptions or contains outliers",
            example: "Comparing methods when data distribution is unknown",
            alternatives: ["Deming Regression (if parametric)", "Kendall's Theil-Sen Regression"]
          } 
        },
        { 
          text: "One method is reference", 
          next: "result", 
          result: { 
            test: "Linear Regression", 
            page: "linear_regression_page", 
            description: "Standard least squares regression",
            assumptions: ["Linear relationship", "Normal residuals", "Homoscedasticity", "Independence"],
            when_to_use: "When one method is considered the reference standard",
            example: "Comparing new method against established gold standard",
            alternatives: ["Weighted Regression", "Robust Regression"]
          } 
        }
      ]
    },
    variance_comparison: {
      question: "How many methods are you comparing?",
      options: [
        { 
          text: "2 methods", 
          next: "result", 
          result: { 
            test: "F-Test for Equal Variances", 
            page: "f_test_page", 
            description: "Compare variances of two groups",
            assumptions: ["Normal distribution in both groups", "Independence of observations"],
            when_to_use: "When comparing precision of two measurement methods",
            example: "Comparing variability of two analytical instruments",
            alternatives: ["Levene's Test (non-normal)", "Brown-Forsythe Test"]
          } 
        },
        { 
          text: "3+ methods", 
          next: "variance_multiple", 
          description: "Multiple variance comparison" 
        }
      ]
    },
    variance_multiple: {
      question: "What are your data characteristics?",
      options: [
        { 
          text: "Normal distribution", 
          next: "result", 
          result: { 
            test: "Bartlett's Test", 
            page: "bartlett_page", 
            description: "Test equal variances assuming normality",
            assumptions: ["Normal distribution in all groups", "Independence of observations"],
            when_to_use: "When testing equal variances across multiple normal groups",
            example: "Comparing precision across multiple laboratories",
            alternatives: ["Levene's Test (robust)", "Brown-Forsythe Test"]
          } 
        },
        { 
          text: "Non-normal or unknown distribution", 
          next: "result", 
          result: { 
            test: "Levene's Test", 
            page: "levene_page", 
            description: "Robust test for equal variances",
            assumptions: ["Independence of observations", "No specific distribution requirements"],
            when_to_use: "When testing equal variances with non-normal data",
            example: "Comparing variability across groups with skewed data",
            alternatives: ["Brown-Forsythe Test", "Bartlett's Test (if normal)"]
          } 
        }
      ]
    },
    distribution_type: {
      question: "What distribution are you testing for?",
      options: [
        { text: "Normal distribution", next: "normality_test", description: "Test if data follows normal distribution" },
        { 
          text: "Specific distribution", 
          next: "result", 
          result: { 
            test: "Kolmogorov-Smirnov Test", 
            page: "ks_test_page", 
            description: "Test fit to any specified distribution",
            assumptions: ["Continuous data", "Independent observations", "Fully specified distribution"],
            when_to_use: "When testing if data follows a specific theoretical distribution",
            example: "Testing if data follows exponential distribution",
            alternatives: ["Anderson-Darling Test", "Cramer-von Mises Test"]
          } 
        },
        { text: "Visual assessment", next: "visual_tests", description: "Graphical methods for distribution assessment" }
      ]
    },
    normality_test: {
      question: "What is your sample size?",
      options: [
        { 
          text: "Small sample (n < 50)", 
          next: "result", 
          result: { 
            test: "Shapiro-Wilk Test", 
            page: "shapiro_wilk_page", 
            description: "Most powerful test for normality with small samples",
            assumptions: ["Sample size ≤ 5000", "Independent observations"],
            when_to_use: "When testing normality with small to moderate sample sizes",
            example: "Testing if laboratory measurements are normally distributed",
            alternatives: ["Anderson-Darling Test", "Q-Q plots for visual assessment"]
          } 
        },
        { 
          text: "Large sample (n ≥ 50)", 
          next: "result", 
          result: { 
            test: "Anderson-Darling Test", 
            page: "anderson_darling_page", 
            description: "Powerful test for normality with larger samples",
            assumptions: ["Independent observations", "Continuous data"],
            when_to_use: "When testing normality with larger sample sizes",
            example: "Testing normality of customer satisfaction scores",
            alternatives: ["Kolmogorov-Smirnov Test", "Shapiro-Wilk Test (if n < 5000)"]
          } 
        }
      ]
    },
    visual_tests: {
      question: "What type of visual assessment do you prefer?",
      options: [
        { 
          text: "Compare quantiles", 
          next: "result", 
          result: { 
            test: "Q-Q Plots", 
            page: "qq_plot_page", 
            description: "Quantile-quantile plots for distribution comparison",
            assumptions: ["Visual interpretation required", "Adequate sample size for patterns"],
            when_to_use: "When visually assessing if data follows a theoretical distribution",
            example: "Checking if residuals from regression are normally distributed",
            alternatives: ["P-P Plots", "Histogram with overlay", "Statistical tests"]
          } 
        },
        { 
          text: "Compare probabilities", 
          next: "result", 
          result: { 
            test: "P-P Plots", 
            page: "pp_plot_page", 
            description: "Probability-probability plots",
            assumptions: ["Visual interpretation required", "Known theoretical distribution"],
            when_to_use: "When comparing cumulative probabilities to theoretical distribution",
            example: "Assessing goodness of fit for survival analysis",
            alternatives: ["Q-Q Plots", "Empirical CDF plots"]
          } 
        }
      ]
    },
    relationship_type: {
      question: "What type of relationship are you analyzing?",
      options: [
        { 
          text: "Linear correlation", 
          next: "correlation_type", 
          description: "Measure strength of linear relationship" 
        },
        { 
          text: "Linear regression", 
          next: "result", 
          result: { 
            test: "Linear Regression", 
            page: "linear_regression_page", 
            description: "Model linear relationships between variables",
            assumptions: ["Linear relationship", "Normal residuals", "Homoscedasticity", "Independence"],
            when_to_use: "When modeling how one variable predicts another",
            example: "Predicting sales based on advertising spend",
            alternatives: ["Polynomial Regression", "Robust Regression", "Non-parametric Regression"]
          } 
        },
        { 
          text: "Non-linear relationship", 
          next: "result", 
          result: { 
            test: "Polynomial/Non-linear Regression", 
            page: "nonlinear_regression_page", 
            description: "Model curved or complex relationships",
            assumptions: ["Adequate sample size", "Correct functional form", "Independence"],
            when_to_use: "When relationship between variables is curved or complex",
            example: "Modeling dose-response curves in pharmacology",
            alternatives: ["Spline Regression", "GAM (Generalized Additive Models)"]
          } 
        }
      ]
    },
    correlation_type: {
      question: "What type of correlation analysis?",
      options: [
        { 
          text: "Continuous variables (parametric)", 
          next: "result", 
          result: { 
            test: "Pearson Correlation", 
            page: "pearson_correlation_page", 
            description: "Measure linear correlation between continuous variables",
            assumptions: ["Linear relationship", "Bivariate normal distribution", "No extreme outliers"],
            when_to_use: "When measuring linear association between two continuous variables",
            example: "Correlation between height and weight",
            alternatives: ["Spearman Correlation (non-parametric)", "Kendall's Tau"]
          } 
        },
        { 
          text: "Non-parametric correlation", 
          next: "result", 
          result: { 
            test: "Spearman Rank Correlation", 
            page: "spearman_correlation_page", 
            description: "Non-parametric measure of monotonic relationship",
            assumptions: ["Monotonic relationship", "Ordinal or continuous data", "Independence"],
            when_to_use: "When data doesn't meet parametric assumptions or relationship is monotonic but not linear",
            example: "Correlation between rankings or skewed data",
            alternatives: ["Kendall's Tau", "Pearson Correlation (if assumptions met)"]
          } 
        }
      ]
    },
    qc_type: {
      question: "What type of quality control analysis?",
      options: [
        { 
          text: "Detect process shifts", 
          next: "result", 
          result: { 
            test: "CUSUM Control Chart", 
            page: "cusum_page", 
            description: "Detect small shifts in process mean",
            assumptions: ["Stable process initially", "Known or estimated parameters", "Independence"],
            when_to_use: "When monitoring for gradual changes in process performance",
            example: "Monitoring laboratory instrument drift over time",
            alternatives: ["Shewhart Control Charts", "EWMA Charts"]
          } 
        },
        { 
          text: "Calculate measurement uncertainty", 
          next: "result", 
          result: { 
            test: "Total Allowable Error Analysis", 
            page: "tae_page", 
            description: "Calculate acceptable measurement error limits",
            assumptions: ["Defined quality requirements", "Known measurement components"],
            when_to_use: "When establishing acceptable limits for measurement systems",
            example: "Setting tolerance limits for clinical laboratory tests",
            alternatives: ["Measurement Uncertainty Evaluation", "Six Sigma Analysis"]
          } 
        },
        { 
          text: "Process capability", 
          next: "result", 
          result: { 
            test: "Process Capability Analysis", 
            page: "capability_page", 
            description: "Assess ability of process to meet specifications",
            assumptions: ["Stable process", "Normal distribution", "Known specification limits"],
            when_to_use: "When evaluating if process can consistently meet requirements",
            example: "Assessing manufacturing process capability",
            alternatives: ["Non-parametric Capability Indices", "Short-term vs Long-term Studies"]
          } 
        }
      ]
    }
  };

  const handleChoice = (option) => {
    const newPath = [...path, { step: currentStep, choice: option.text }];
    setPath(newPath);
    
    if (option.result) {
      setCurrentStep('result');
    } else {
      setCurrentStep(option.next);
    }
  };

  const resetSelection = () => {
    setCurrentStep('start');
    setPath([]);
  };

  const goBack = () => {
    if (path.length > 0) {
      const newPath = [...path];
      const lastStep = newPath.pop();
      setPath(newPath);
      
      if (newPath.length === 0) {
        setCurrentStep('start');
      } else {
        const previousPath = [...newPath];
        let currentNode = 'start';
        for (const pathStep of previousPath) {
          const node = decisionTree[currentNode];
          const selectedOption = node?.options?.find(opt => opt.text === pathStep.choice);
          if (selectedOption) {
            currentNode = selectedOption.next;
          }
        }
        setCurrentStep(currentNode);
      }
    }
  };

  const getCurrentResult = () => {
    if (path.length === 0) return null;
    
    let currentNode = 'start';
    let result = null;
    
    for (const pathStep of path) {
      const node = decisionTree[currentNode];
      const selectedOption = node?.options?.find(opt => opt.text === pathStep.choice);
      if (selectedOption) {
        if (selectedOption.result) {
          result = selectedOption.result;
        }
        currentNode = selectedOption.next;
      }
    }
    
    return result;
  };

  const currentNode = decisionTree[currentStep];
  const result = currentStep === 'result' ? getCurrentResult() : null;

  return (
    <div className="max-w-5xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-2">
            <BookOpen className="text-blue-600" />
            Statistical Test Selector
          </h1>
        </div>
        <p className="text-gray-600 mb-8">
          Answer a few simple questions to find the right statistical test for your data analysis.
        </p>
        
        {path.length > 0 && (
          <div className="mb-6">
            <div className="flex items-center text-sm text-gray-500 flex-wrap">
              {path.map((step, index) => (
                <React.Fragment key={index}>
                  <span className="font-medium text-blue-600">{step.choice}</span>
                  {index < path.length - 1 && <ChevronRight className="w-4 h-4 mx-1 text-gray-400" />}
                </React.Fragment>
              ))}
            </div>
            <div className="flex gap-4 mt-4">
              <button 
                onClick={goBack} 
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 flex items-center transition-colors"
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                Go Back
              </button>
              <button
                onClick={resetSelection}
                className="px-4 py-2 text-red-600 border border-red-200 rounded-lg text-sm font-medium hover:bg-red-50 transition-colors"
              >
                Start Over
              </button>
            </div>
          </div>
        )}

        {currentStep !== 'result' ? (
          <div>
            <h2 className="text-xl font-semibold text-gray-700 mb-4">{currentNode?.question}</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {currentNode?.options?.map((option, index) => (
                <button
                  key={index}
                  onClick={() => handleChoice(option)}
                  className="group flex flex-col items-start p-6 bg-gray-50 border border-gray-200 rounded-lg text-left transition-all duration-200 hover:bg-blue-50 hover:border-blue-400 hover:shadow-md"
                >
                  <span className="text-lg font-medium text-gray-800 group-hover:text-blue-600 transition-colors">
                    {option.text}
                  </span>
                  <p className="mt-2 text-sm text-gray-500 group-hover:text-blue-700 transition-colors">
                    {option.description}
                  </p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="bg-green-50 border-l-4 border-green-500 text-green-800 p-6 rounded-lg mb-8">
            <div className="flex items-center mb-4">
              <CheckCircle className="w-6 h-6 mr-3 text-green-600" />
              <h2 className="text-2xl font-bold">Suggested Test: <span className="text-green-700">{result?.test}</span></h2>
            </div>
            <p className="text-lg mb-4">{result?.description}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded-lg border border-green-200">
                <h3 className="flex items-center text-lg font-semibold text-gray-700 mb-2">
                  <AlertCircle className="w-5 h-5 text-yellow-500 mr-2" />
                  Key Assumptions
                </h3>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  {result?.assumptions?.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>
              <div className="bg-white p-4 rounded-lg border border-green-200">
                <h3 className="flex items-center text-lg font-semibold text-gray-700 mb-2">
                  <Info className="w-5 h-5 text-blue-500 mr-2" />
                  When to Use
                </h3>
                <p className="text-gray-600">{result?.when_to_use}</p>
              </div>
              <div className="bg-white p-4 rounded-lg border border-green-200 col-span-1 md:col-span-2">
                <h3 className="flex items-center text-lg font-semibold text-gray-700 mb-2">
                  <BookOpen className="w-5 h-5 text-purple-500 mr-2" />
                  Example
                </h3>
                <p className="text-gray-600">{result?.example}</p>
              </div>
            </div>
            {result?.alternatives && result.alternatives.length > 0 && (
              <div className="bg-white p-4 rounded-lg border border-green-200 mt-4">
                <h3 className="flex items-center text-lg font-semibold text-gray-700 mb-2">
                  <RotateCcw className="w-5 h-5 text-red-500 mr-2" />
                  Alternatives
                </h3>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  {result?.alternatives?.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default StatisticalTestSelector;