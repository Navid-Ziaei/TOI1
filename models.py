from LDGD.model import *
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from LDGD import visualization
from imblearn.over_sampling import SMOTE
import torch
import xgboost as xgb
def train_xgb(data_train, labels_train, data_test, labels_test, paths, balance_method='smote'):
    # Create and train the XGBoost model with class weights

    if len(np.unique(labels_train)) > 2:
        model = xgb.XGBClassifier(objective="multi:softmax", num_class=2)
    else:
        scale_pos_weight = 1
        if balance_method == 'smote':
            smote = SMOTE(random_state=42)
            data_train, labels_train = smote.fit_resample(data_train, labels_train)
        elif balance_method == 'weighting':
            scale_pos_weight = 2 * np.sum(1 - labels_train) / np.sum(labels_train)

        model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                  max_depth=6,
                                  num_parallel_tree=2)

    model.fit(data_train, labels_train)

    # Make predictions
    predictions = model.predict(data_test)

    # Calculate the F1-score
    f1 = f1_score(labels_test, predictions, average='macro')
    print(f"F1-score: {f1 * 100:.2f}%")

    report = classification_report(y_true=labels_test, y_pred=predictions)
    print(report)
    metrics = {
        'accuracy': accuracy_score(labels_test, predictions),
        'precision': precision_score(labels_test, predictions, average='weighted'),
        'recall': recall_score(labels_test, predictions, average='weighted'),
        'f1_score': f1_score(labels_test, predictions, average='weighted')
    }

    with open(paths.path_result + 'xgb_classification_report.txt', "w") as file:
        file.write(report)

    with open(paths.path_result + 'xgb_classification_result.json', "w") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def train_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test,
               settings, paths,
               shared_inducing_points=False,
               use_shared_kernel=False,
               cls_weight=1.0,
               reg_weight=1.0,
               early_stop=None):
    settings.data_dim = data_train.shape[-1]
    batch_shape = torch.Size([settings.data_dim])
    if settings.use_gpytorch is False:
        kernel_cls = ARDRBFKernel(input_dim=settings.latent_dim)
        kernel_reg = ARDRBFKernel(input_dim=settings.latent_dim)
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=settings.latent_dim))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=settings.latent_dim))

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    y_train_onehot = torch.tensor(y_train)
    y_test_onehot = torch.tensor(y_test)

    model = LDGD(torch.tensor(data_train, dtype=torch.float32),
                                kernel_reg=kernel_reg,
                                kernel_cls=kernel_cls,
                                num_classes=y_train_onehot.shape[-1],
                                latent_dim=settings.latent_dim,
                                num_inducing_points_cls=settings.num_inducing_points,
                                num_inducing_points_reg=settings.num_inducing_points,
                                likelihood_reg=likelihood_reg,
                                likelihood_cls=likelihood_cls,
                                use_gpytorch=settings.use_gpytorch)
    # shared_inducing_points=shared_inducing_points,
    # use_shared_kernel=use_shared_kernel,
    # cls_weight=cls_weight,
    # reg_weight=reg_weight)

    if settings.load_trained_model is False:
        losses, *_ = model.train_model(yn=data_train, ys=y_train_onehot,
                                       epochs=settings.num_epochs_train,
                                       batch_size=settings.batch_size)
        # early_stop=early_stop)
        model.save_wights(path_save=paths.path_model)

    else:
        losses = []
        model.load_weights(paths.path_model)

    predictions, metrics, *_ = model.evaluate(yn_test=data_test, ys_test=labels_test,
                                              epochs=settings.num_epochs_test,
                                              save_path=paths.path_result)

    if settings.use_gpytorch is False:
        alpha_reg = model.kernel_reg.alpha.detach().cpu().numpy()
        alpha_cls = model.kernel_cls.alpha.detach().cpu().numpy()
        X = model.x.q_mu.detach().cpu().numpy()
        std = model.x.q_sigma.detach().cpu().numpy()
    else:
        alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.detach().cpu().numpy()
        alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.detach().cpu().numpy()
        X = model.x.q_mu.detach().cpu().numpy()
        std = torch.nn.functional.softplus(model.x.q_log_sigma).detach().cpu().numpy()

    visualization.plot_results_gplvm(X, np.sqrt(std), labels=labels_train, losses=losses,
                                     inverse_length_scale=alpha_reg,
                                     latent_dim=settings.latent_dim,
                                     save_path=paths.path_result, file_name=f'gplvm_train_reg_result_all',
                                     show_errorbars=True)
    visualization.plot_results_gplvm(X, np.sqrt(std), labels=labels_train, losses=losses,
                                     inverse_length_scale=alpha_cls,
                                     latent_dim=settings.latent_dim,
                                     save_path=paths.path_result, file_name=f'gplvm_train_cls_result_all',
                                     show_errorbars=True)

    if settings.latent_dim is False:

        X_test = model.x_test.q_mu.detach().cpu().numpy()
        std_test = model.x_test.q_sigma.detach().cpu().numpy()
    else:

        X_test = model.x_test.q_mu.detach().cpu().numpy()
        std_test = torch.nn.functional.softplus(model.x_test.q_log_sigma).detach().cpu().numpy()
    visualization.plot_results_gplvm(X_test, std_test, labels=labels_test, losses=losses,
                                     inverse_length_scale=alpha_cls,
                                     latent_dim=settings.latent_dim,
                                     save_path=paths.path_result, file_name=f'gplvm_test_result_all',
                                     show_errorbars=True)

    """inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])

    plot_heatmap(X, labels_train, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_train', inducing_points=inducing_points, save_path=paths.path_result[0])
    plot_heatmap(X_test, labels_test, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_test', inducing_points=inducing_points, save_path=paths.path_result[0])"""

    return metrics


def train_fast_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test,
               settings, paths,
               shared_inducing_points=False,
               use_shared_kernel=False,
               cls_weight=1.0,
               reg_weight=1.0,
               early_stop=None):
    settings.data_dim = data_train.shape[-1]
    batch_shape = torch.Size([settings.data_dim])
    if settings.use_gpytorch is False:
        kernel_cls = ARDRBFKernel(input_dim=settings.latent_dim)
        kernel_reg = ARDRBFKernel(input_dim=settings.latent_dim)
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=settings.latent_dim))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=settings.latent_dim))

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    y_train_onehot = torch.tensor(y_train)
    y_test_onehot = torch.tensor(y_test)

    model = FastLDGD(torch.tensor(data_train, dtype=torch.float32),
                                kernel_reg=kernel_reg,
                                kernel_cls=kernel_cls,
                                num_classes=y_train_onehot.shape[-1],
                                latent_dim=settings.latent_dim,
                                num_inducing_points_cls=settings.num_inducing_points,
                                num_inducing_points_reg=settings.num_inducing_points,
                                likelihood_reg=likelihood_reg,
                                likelihood_cls=likelihood_cls,
                                use_gpytorch=settings.use_gpytorch)
    # shared_inducing_points=shared_inducing_points,
    # use_shared_kernel=use_shared_kernel,
    # cls_weight=cls_weight,
    # reg_weight=reg_weight)

    if settings.load_trained_model is False:
        losses, *_ = model.train_model(yn=data_train, ys=y_train_onehot,
                                       epochs=settings.num_epochs_train,
                                       batch_size=settings.batch_size)
        # early_stop=early_stop)
        model.save_wights(path_save=paths.path_model)

    else:
        losses = []
        model.load_weights(paths.path_model)

    predictions, metrics, *_ = model.evaluate(yn_test=data_test, ys_test=labels_test,
                                              epochs=settings.num_epochs_test,
                                              save_path=paths.path_result)

    return metrics