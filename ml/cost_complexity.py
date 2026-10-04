from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from .model_data import (
    prepare_data,
    classification_metrics
)

_CCP_CACHE = None


def run_cost_complexity(force=False):
    global _CCP_CACHE
    if not force and _CCP_CACHE is not None:
        return _CCP_CACHE

    (
        X_train,
        X_test,
        y_train,
        y_test,
        pre
    ) = prepare_data(
        scale_numeric=False
    )

    # Validation split
    (
        X_part,
        X_val,
        y_part,
        y_val
    ) = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    Xp = pre.fit_transform(X_part)
    Xv = pre.transform(X_val)

    # Initial constrained tree for fast pruning path
    base = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        random_state=42
    )

    base.fit(Xp, y_part)

    path = base.cost_complexity_pruning_path(Xp, y_part)
    alphas = path.ccp_alphas

    # Sample up to 10 alphas for fast, robust cross-validation
    if len(alphas) > 10:
        step = len(alphas) // 10
        sampled_alphas = alphas[::step][:10]
    else:
        sampled_alphas = alphas

    best_alpha = float(alphas[0])
    best_score = -1.0

    for alpha in sampled_alphas:
        tree = DecisionTreeClassifier(
            random_state=42,
            max_depth=15,
            ccp_alpha=float(alpha)
        )
        tree.fit(Xp, y_part)
        score = tree.score(Xv, y_val)

        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    # Final model on full training set
    pre.fit(X_train)
    Xt = pre.transform(X_train)
    Xte = pre.transform(X_test)

    tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=15,
        ccp_alpha=best_alpha
    )

    tree.fit(Xt, y_train)
    pred = tree.predict(Xte)

    metrics = classification_metrics(y_test, pred)
    metrics.update({
        "model": "Cost-Complexity Pruning",
        "ccp_alpha": best_alpha,
        "tree_depth": tree.get_depth(),
        "leaf_nodes": tree.get_n_leaves()
    })

    _CCP_CACHE = metrics
    return metrics