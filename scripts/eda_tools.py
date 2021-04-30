from functools import partial

from dtreeviz.trees import dtreeviz
from matplotlib.pyplot import show, subplots
import matplotlib.pyplot as plt
from numpy import corrcoef
import pandas as pd
import seaborn as sns
from seaborn import barplot
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier


def count_and_normalize(data, feature_1, feature_2):

    # First we group and count the features.
    grouped_by_features = (
        data
        .groupby([feature_1, feature_2], as_index=False)
        .agg(count=(feature_2, 'count'))
    )

    # Then we get the total number of ocurrences for each one.
    total_counts = (
        grouped_by_features
        .groupby(feature_1, as_index=False)
        .agg(total_counts=('count', 'sum'))
    )

    # Put counts and totals together.
    grouped_with_total = (
        grouped_by_features
        .merge(
            total_counts,
            on=feature_1,
        )
    )

    # Normalize by dividing counts by total.
    grouped_with_fraction = (
        grouped_with_total
        .add_column(
            'fraction',
            grouped_with_total['count'] / grouped_with_total['total_counts']
        )
        .drop(columns=['count', 'total_counts'])
        .set_index([feature_1, feature_2])
    )

    return grouped_with_fraction


def check_correlation(data, feature_1, feature_2):

    _, axes = subplots(1, 2, figsize=(12, 5))

    grouped_with_fraction_1 = count_and_normalize(data, feature_1, feature_2).reset_index()
    barplot(
        data=grouped_with_fraction_1,
        x=feature_1,
        y='fraction',
        hue=feature_2,
        ax=axes[0]
    )

    grouped_with_fraction_2 = count_and_normalize(data, feature_2, feature_1).reset_index()
    barplot(
        data=grouped_with_fraction_2,
        x=feature_2,
        y='fraction',
        hue=feature_1,
        ax=axes[1]
    )

    correlation = corrcoef(
        data[[feature_1, feature_2]].to_numpy().T
    )[0, 1]

    print(
        f'The correlation coefficient between\n"{feature_1}" and "{feature_2}"\nis {correlation}.'
    )

    show()


def norm_barplot(data, feature):

    normalized_counts = data[feature].value_counts(normalize=True)

    bar = barplot(
        x=normalized_counts.index,
        y=normalized_counts.to_numpy(),
    )

    bar.set_xlabel(feature)
    bar.set_ylabel('fraction')

    counts = (
        data[feature]
        .value_counts()
        .rename('counts')
        .to_frame()
    )

    norm_counts = (
        data[feature]
        .value_counts(normalize=True)
        .rename('normalized_counts')
        .to_frame()
    )

    df = pd.concat([counts, norm_counts], axis=1).rename_axis(feature, axis=0)
    
    # print(df)
    return df

    show()


def report(y_true, y_pred):
    """Report relevant training results: sensitivity, specificity and area under ROC curve."""
    conf_matrix = confusion_matrix(y_true, y_pred)

    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]
    fn = conf_matrix[1][0]
    fp = conf_matrix[0][1]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    report = {
        "sensitivity": [sensitivity],
        "specificity": [specificity],
        "roc_auc": [roc_auc_score(y_true, y_pred)],
    }

    return report


def create_tree(df, max_depth, target='admitted', **kwargs):

    X = df.drop(columns=[target])
    y = df[target]

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        **kwargs,
    )

    tree.fit(X, y)

    rep = report(y, tree.predict(X))

    viz = dtreeviz(
        tree,
        X,
        y,
        target_name='admitted',
        feature_names=X.columns,
        class_names=['no', 'yes'],
    )

    return tree, rep, viz

def impute_missing_values(df, aggfunc='mean', cols_to_ignore=None, categorical_cols=None, n_neighbors=2, metric='minkowski', p=2, n_jobs=-1):

    # Target variable and any other feature that should not be used for imputation.
    if cols_to_ignore is not None:
        df_imp_feats = df.drop(columns=cols_to_ignore)
        df_ignored_cols = df[cols_to_ignore]
    else:
        df_imp_feats = df
        df_ignored_cols = None

    # Only the rows with no missing values will be used for imputation.
    training_set = df_imp_feats.dropna(how='any', axis=0)
    # The remaining rows, which have missing values, are the ones for which we're imputing.
    rows_to_impute = df_imp_feats.drop(index=training_set.index)

    neigh = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
    )

    # This pd.DataFrame skeleton will be used to concatenate the imputed rows.
    partial_imputed_df = pd.DataFrame(columns=df_imp_feats.columns)
    partial_imputed_df.index.name = df_imp_feats.index.name

    # Remember iterrows yields a (pd.Index, pd.Series) tuple.
    for _, row in rows_to_impute.iterrows():

        missing_features = row[row.isna()].index

        # In this context, the projection means every feature that is present in the
        # assessed row. In other words, just ignore features for which the value is
        # missing.
        projected_training_set = training_set.drop(columns=missing_features)
        # Since row is a pd.Series, we have to convert it to pd.DataFrame. The transposition
        # makes sure the features become columns names.
        projected_row_to_impute = (
            row
            .drop(missing_features)
            .to_frame()
            .T
        )

        neigh.fit(projected_training_set)
        closest_neighbors_idx = neigh.kneighbors(
            projected_row_to_impute,
            n_neighbors=n_neighbors,
            return_distance=False,
        )

        closest_neighbors_df = (
            training_set
            # The kneighbors method does not return the row label, but the actual array
            # index, so we use iloc to select the rows and then loc to get the features.
            .iloc[closest_neighbors_idx[0]]
            .loc[:, missing_features]
        )

        def agg_from_neighbors(column, categorical_cols):
            if column.name in categorical_cols:
                return column.agg('mode')[0]
            else:
                return column.agg(aggfunc)

        closest_neighbors_agg = closest_neighbors_df.apply(
            partial(
                agg_from_neighbors,
                categorical_cols=categorical_cols
            )
        )

        row = row.to_frame().T
        row.index.name = df_imp_feats.index.name
        for feature in closest_neighbors_agg.index:
            row[feature] = closest_neighbors_agg[feature]
        partial_imputed_df = partial_imputed_df.append(row)

    imputed_df_all_rows = pd.concat([training_set, partial_imputed_df])

    if df_ignored_cols is not None:
        final_imputed_df = pd.merge(
            df_ignored_cols,
            imputed_df_all_rows,
            on='id',
        )
    else:
        final_imputed_df = imputed_df_all_rows

    return final_imputed_df

def compare_imputation_methods(original_df, imputed_df, feature, ax=None):
    comparison = pd.concat(
        [
            original_df[feature],
            imputed_df[feature],
            (
                original_df[feature]
                .fillna(
                    value=original_df[feature]
                    .mean()
                )
            ),
        ],
        axis=1
    )

    comparison.columns = ['original', 'knn', 'mean']

    sns.boxplot(
        data=comparison,
        ax=ax
    )

    if ax is None:
        plt.title(feature)
    else:
        ax.set_title(feature)
