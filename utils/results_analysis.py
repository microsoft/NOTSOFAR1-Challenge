import pandas as pd
from pathlib import Path
from inference_pipeline.load_meeting_data import load_data
import numpy as np
from scipy.stats import t
from typing import Union, Tuple
import matplotlib.pyplot as plt


class ResultsAnalyser:
    def __init__(self, all_session_wer_df: pd.DataFrame, all_gt_metadata_df: pd.DataFrame,
                 all_session_wer_ref_df: pd.DataFrame = None):

        """
        Initializes the ResultsAnalyser with DataFrames containing all sessions WER (i.e., results) and all ground truth
        metadata. If all_session_wer_ref_df is not provided, metadata_analysis() is evaluated on the results provided in
        all_session_wer_df. If all_session_wer_ref_df is provided, metadata_analysis() is evaluated on the difference
        between the results, i.e., all_session_wer_ref_df - all_session_wer_df.
        When comparing two systems, it is recommended to evaluate the difference between the two systems, since it
        accounts for the relative performance of the two systems rather than their absolute performance, which leads to
        smaller variance (for the difference) and hence tighter confidence intervals (CIs). Note that to claim that
        all_session_wer_df is significantly better than all_session_wer_ref_df, the CI of the difference should not
        contain 0.


        Args:
            all_session_wer_df: The DataFrame containing all session WER data.
            all_gt_metadata_df: The DataFrame containing all ground truth metadata.
            all_session_wer_ref_df: The DataFrame containing all session WER data of a reference model. If provided, the
                                    mean of the difference is evaluated, and the CIs accordingly.
        """

        self.all_session_wer_df = all_session_wer_df
        self.all_gt_metadata_df = all_gt_metadata_df
        self.all_session_wer_df['meeting_id'] = self.all_session_wer_df['session_id'].str.extract(r'(MTG_\d+)')
        self.all_session_wer_with_metadata_df = pd.merge(self.all_session_wer_df, self.all_gt_metadata_df,
                                                         on='meeting_id')

        self.all_session_wer_ref_df = all_session_wer_ref_df
        if all_session_wer_ref_df is not None:
            assert all_session_wer_ref_df['session_id'].equals(all_session_wer_df['session_id']), \
                   "Tested and reference results must be aligned in the sessions column."
            self.all_session_wer_ref_df['meeting_id'] = self.all_session_wer_ref_df['session_id'].str.extract(r'(MTG_\d+)')
            self.all_session_wer_ref_with_metadata_df = pd.merge(self.all_session_wer_ref_df, self.all_gt_metadata_df,
                                                                 on='meeting_id')

    def metadata_analysis(self, gather_near_whiteboard: bool = True, verbose: bool = False,min_samples_for_ci: int = 20,
                          confidence_level: float = 0.95, ci_over_sessions: bool = False, bootstrap_samples: int = 0,
                          plot_results: bool = False):

        """
        Analyzes the results given the metadata, i.e., the results for all items (sessions or meetings) as well as
        conditioned on metadata hashtags, and calculates confidence intervals for each (if the min_samples_for_ci
        criteria is met, see below).

        Args:
            gather_near_whiteboard: If True, gathers data for sessions in which one of the participants is near the
                                    whiteboard.
            verbose: If True, prints the mean tcp_wer and tcorc_wer for each segment/hashtag.
            min_samples_for_ci: The minimum number of samples required to calculate the confidence interval. If the
                                number of samples is less than this value, the confidence interval is set to (NaN, NaN).
            confidence_level: The confidence level (>=0 and <1) of the confidence interval for the mean WER values.
                              If 0, no confidence interval is calculated.
            ci_over_sessions: If True, calculates the confidence interval over all sessions. If False, calculates the
                              confidence interval over meeting (which are more likely to be i.i.d. than sessions).
            bootstrap_samples: The number of bootstrap samples to use for calculating the confidence interval. If set
                               to 0 (default), no bootstrapping is performed.
            plot_results: If True, plots the results of the metadata analysis.

        Returns:
            DataFrame: A DataFrame containing the analysis results. The WER values are between 0 and 1.

            An example of a returned dataset (with min_samples_for_ci=20):

                                    tcp_wer	tcp_wer_ci	        tcorc_wer	tcorc_wer_ci	    confidence_level	len
        all_items                   0.32476	(0.30716, 0.3425)	0.26764	    (0.25556, 0.28094)	0.95	            106
        #NaturalMeeting             0.32328	(nan, nan)	        0.26256	    (nan, nan)	        0.95	            18
        #TalkNearWhiteboard=Ernie   0.46548	(nan, nan)	        0.20229	    (nan, nan)	        0.95	            3
        #DebateOverlaps             0.37969	(0.35631, 0.4039)	0.31441	    (0.28802, 0.33709)	0.95	            24
        #LeaveAndJoin=Ernie         0.27533	(nan, nan)	        0.2508	    (nan, nan)	        0.95	            2
        ...


        """
        assert 0 <= confidence_level < 1, "Confidence level must be between 0 and 1."

        df_to_analyze = self.all_session_wer_with_metadata_df.copy() # avoid changing the original df.
        # if self.all_session_wer_ref_df is not None then calculate the difference between the two models
        if self.all_session_wer_ref_df is not None:
            df_to_analyze['tcp_wer'] = self.all_session_wer_ref_with_metadata_df['tcp_wer'] - \
                                       self.all_session_wer_df['tcp_wer']
            df_to_analyze['tcorc_wer'] = self.all_session_wer_ref_with_metadata_df['tcorc_wer'] - \
                                         self.all_session_wer_df['tcorc_wer']

        if not ci_over_sessions:
            # Group by 'meeting_id'
            grouped = df_to_analyze.groupby('meeting_id', group_keys=False)

            # For numeric columns, calculate the mean
            numeric_cols = df_to_analyze.select_dtypes(include=[np.number]).columns
            numeric_df = grouped[numeric_cols].mean()

            # We only use 'Hashtags' non-numeric column in our analysis, hence we first assert that all entries in the
            # 'Hashtags' column are the same for each meeting, and then select a random entry For non-numeric columns.
            assert grouped['Hashtags'].nunique().eq(1).all(), "All sessions' 'Hashtags' for each meeting must be the same."
            non_numeric_cols = df_to_analyze.select_dtypes(exclude=[np.number]).columns
            non_numeric_df = grouped[non_numeric_cols].apply(lambda x: x.sample(1).iloc[0])

            # Concatenate the results
            all_items_wer_with_metadata_df = pd.concat([numeric_df, non_numeric_df], axis=1)
        else:
            # Use the original DataFrame as it contains sessions
            all_items_wer_with_metadata_df = df_to_analyze

        if len(self.all_session_wer_df["tcp_wer"]) >= min_samples_for_ci:
            tcp_wer_ci = calculate_confidence_interval_of_mean(all_items_wer_with_metadata_df["tcp_wer"],
                                                               confidence_level, bootstrap_samples=bootstrap_samples)
            tcorc_wer_ci = calculate_confidence_interval_of_mean(all_items_wer_with_metadata_df["tcorc_wer"],
                                                                 confidence_level, bootstrap_samples=bootstrap_samples)
        else:
            tcp_wer_ci = (np.nan, np.nan)
            tcorc_wer_ci = (np.nan, np.nan)

        results = {'all_items':
                       {'tcp_wer': all_items_wer_with_metadata_df["tcp_wer"].mean(),
                        'tcp_wer_ci': tcp_wer_ci,
                        'tcorc_wer': all_items_wer_with_metadata_df["tcorc_wer"].mean(),
                        'tcorc_wer_ci': tcorc_wer_ci,
                        'confidence_level': confidence_level,
                        'len': len(all_items_wer_with_metadata_df)}
                   }

        if verbose:
            print(f'mean tcp_wer of all items ({len(all_items_wer_with_metadata_df)}) = '
                  f'{results["all_items"]["tcp_wer"]}')
            print(f'mean tcorc_wer of all items ({len(all_items_wer_with_metadata_df)}) = '
                  f'{results["all_items"]["tcorc_wer"]}')

        unique_hashtags = all_items_wer_with_metadata_df['Hashtags'].str.split(', ').explode().unique()
        if gather_near_whiteboard:
            unique_hashtags = np.append(unique_hashtags, '#TalkNearWhiteboard')
        for hashtag in unique_hashtags:

            if hashtag == '#TalkNearWhiteboard':
                hashtag_items = all_items_wer_with_metadata_df[
                    all_items_wer_with_metadata_df['Hashtags'].str.contains(hashtag, regex=False)]
            else:
                hashtag_items = all_items_wer_with_metadata_df[
                    all_items_wer_with_metadata_df['Hashtags'].str.split(', ').apply(lambda x: hashtag in x)]

            if len(hashtag_items) >= min_samples_for_ci:
                tcp_wer_ci = calculate_confidence_interval_of_mean(hashtag_items["tcp_wer"], confidence_level,
                                                                   bootstrap_samples=bootstrap_samples)
                tcorc_wer_ci = calculate_confidence_interval_of_mean(hashtag_items["tcorc_wer"], confidence_level,
                                                                     bootstrap_samples=bootstrap_samples)
            else:
                tcp_wer_ci = (np.nan, np.nan)
                tcorc_wer_ci = (np.nan, np.nan)


            results[hashtag] = {
                'tcp_wer': hashtag_items["tcp_wer"].mean(),
                'tcp_wer_ci': tcp_wer_ci,
                'tcorc_wer': hashtag_items["tcorc_wer"].mean(),
                'tcorc_wer_ci': tcorc_wer_ci,
                'confidence_level': confidence_level,
                'len': len(hashtag_items)}

            if verbose:
                print(f'mean tcp_wer of {hashtag} ({len(hashtag_items)})= {results[hashtag]["tcp_wer"]}')
                print(f'mean tcorc_wer of {hashtag} ({len(hashtag_items)})= {results[hashtag]["tcorc_wer"]}')

        results_df = pd.DataFrame.from_dict(results, orient='index')
        if plot_results:
            plot_metadata_results(results_df)
        return results_df


def calculate_confidence_interval_of_mean(data: Union[np.ndarray, list, pd.Series], confidence_level: float,
                                          bootstrap_samples: int = 0):
    """
    Calculates the confidence interval of the mean for a given data set of i.i.d. samples and a confidence level.
    The function supports two methods for calculating the confidence interval of the mean:
    1. Student's t-distribution: This method is appropriate when the sample size is small and/or the population
       standard deviation is unknown. It assumes that the data follows a normal distribution or the sample size is
       large enough for the Central Limit Theorem to hold. This method is used when `bootstrap_samples` is set to 0.
    2. Bootstrapping: This is a non-parametric method that makes no assumptions about the data distribution. It can
       be used when the sample size is small and the data does not follow a normal distribution. Note that although
       valid in small sample size, bootstraping doesn't compensate on a lack of data.
       This method is used when `bootstrap_samples` is greater than 0.

    Args:
        data: The data for which the confidence interval is to be calculated.
        confidence_level: The confidence level for the confidence interval. This should be a value between 0 and 1. For
                          example, a confidence level of 0.95 corresponds to a 95% confidence interval.
        bootstrap_samples: The number of bootstrap samples to use for calculating the confidence interval. If set to
                            0 (default), no bootstrapping is performed and the t-distribution is used instead. Note that
                            bootstrapping does not necessarily produce symmetric confidence intervals.

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval of the mean.
    """

    # Use bootstrapping to calculate confidence interval
    if bootstrap_samples > 0:
        bootstrap_means = []
        for _ in range(bootstrap_samples):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        lower_percentile = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
        upper_percentile = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)
        confidence_interval = (lower_percentile, upper_percentile)

    # Use t-distribution to calculate confidence interval
    else:
        # Calculate sample statistics
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)  # ddof=1 for *sample* standard deviation
        sample_size = len(data)
        df = sample_size - 1
        # Calculate confidence interval using scipy.stats.t.interval
        confidence_interval = t.interval(confidence_level, df, loc=sample_mean, scale=sample_std / np.sqrt(sample_size))

    return confidence_interval


def plot_metadata_results(results_df: pd.DataFrame, fig_size: Tuple = (20, 12), font_size: int = 20):
    """
    Plots the results of the metadata analysis.

    Args:
        results_df: The DataFrame containing the results of the metadata analysis.
        fig_size: The size of the figure. Default is (20, 12).
        font_size: The font size of the labels. Default is 20.
    """
    fig, ax = plt.subplots(1, 2, figsize=fig_size)
    fz = font_size
    for i, wer_type in enumerate(['tcp_wer', 'tcorc_wer']):
        means = results_df[wer_type]
        confidence_intervals = results_df[f'{wer_type}_ci']
        errors = [means - [ci[0] for ci in confidence_intervals], [ci[1] for ci in confidence_intervals] - means]
        for j in range(len(results_df.index)):
            color = 'red' if np.isnan(errors[0][j]) else 'blue'
            ax[i].errorbar(results_df.index[j], means[j], yerr=[[errors[0][j]], [errors[1][j]]], fmt='o', color=color)
        ax[i].set_xlabel('Hashtag', fontsize=fz)
        ax[i].set_ylabel(f'Mean {wer_type}', fontsize=fz)
        ax[i].set_title(f'CI of mean {wer_type} per hashtag', fontsize=fz)
        for label in ax[i].get_xticklabels():
            label.set_rotation(90)
            label.set_fontsize(fz)
        for label in ax[i].get_yticklabels():
            label.set_fontsize(fz)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    meetings_dir = Path('../artifacts/meeting_data/dev_set/240415.2_dev_with_GT/MTG')
    _, _, all_gt_metadata_df = load_data(str(meetings_dir), None)

    exp_dir = '../exp_name/'
    results_tsv_path = Path(exp_dir) / 'artifacts/outputs/wer/css_large-v3_word_nmesc_results.csv'
    all_session_wer_df = pd.read_csv(results_tsv_path, delimiter='\t')

    # Uncomment the following lines to compare the results with reference results, e.g., of a baseline model.
    # ref_exp_dir = '../ref_exp_name/'
    # results_ref_tsv_path = Path(ref_exp_dir) / 'artifacts/outputs/wer/css_large-v3_word_nmesc_results.csv'
    # all_session_wer_ref_df = pd.read_csv(results_ref_tsv_path, delimiter='\t')

    res_analyser = ResultsAnalyser(all_session_wer_df=all_session_wer_df, all_gt_metadata_df=all_gt_metadata_df)
                                   #all_session_wer_ref_df=all_session_wer_ref_df)
    results = res_analyser.metadata_analysis(gather_near_whiteboard=True, confidence_level=0.95,
                                             bootstrap_samples=0, min_samples_for_ci=20, ci_over_sessions=False)
