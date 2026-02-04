import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import torch, gpytorch
import numpy as np
from GPassets.preprocessing import PreProcessor
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import time
from datetime import datetime 

class GPModel:
    """
    Python class for Gaussian Process modelling
    """
    def __init__(self, config : dict, prep : PreProcessor = None, device : str = 'cuda'):
       
        self.config = config # Must contain fields 'kernel', 'version'
        self.prep = prep 
        self.x_train = None
        self.y_train = None 
        self.x_test = None 
        self.y_test = None 
        self.model = None 
        self.train_report = None
        self.test_report = None 
        self.device = device 
    
    # ---------------------------- Personal Helpers -----------------------------------------------------
    @torch.no_grad()
    def _eval_loss(self, mll) -> float:
        """
        Internal method to manually evaluate loss after iteration
        
        :param mll: Maximum likelihood object
        :return: loss value
        :rtype: float
        """
        out = self.model(self.model.train_inputs[0])
        loss = -mll(out, self.model.train_targets).sum()
        return float(loss.detach().cpu().item())
    
    @torch.no_grad()
    def _get_hypers(self):
        """
        Internal method for extracting hyperparameters
        """
        out = {}

        # Likelihood noise covariance
        noise = self.model.likelihood.noise
        out['noise_covariance'] = self._summary(noise)
        # Constant mean
        mean = self.model.mean_module.constant
        out["mean_constant"] = self._summary(mean)

        # outputscale (ScaleKernel)
        out["outputscale"] = self._summary(self.model.covar_module.outputscale)

        kt = self.config['kernel'].casefold()
        
        if kt == "matern12".casefold():
            ls = self.model.covar_module.base_kernel.lengthscale 
            out["lengthscale"] = self._summary(ls)

        elif kt == "relu".casefold():
            out["weight_var"] = self._summary(self.model.covar_module.base_kernel.weight_var)
            out["bias_var"] = self._summary(self.model.covar_module.base_kernel.bias_var)
        
        return out
    
    @staticmethod
    def _summary(values):
        """
        Summarize a scalar or vector as python floats.
        Returns:
          - {"val": x} for scalars
          - {"min": a, "median": b, "max": c, "len": n} for vectors
        """
        # Convert to 1D torch tensor on CPU
        if isinstance(values, torch.Tensor):
            t = values.detach().flatten().cpu().float()
        else:
            # values could be python float/int or list/np array
            arr = np.asarray(values)
            t = torch.as_tensor(arr).flatten().cpu().float()

        if t.numel() == 1:
            return {"val": float(t.item())}

        # torch.median returns one of the middle values for even length (fine for reporting)
        return {
            "min": float(t.min().item()),
            "median": float(t.median().item()),
            "max": float(t.max().item())
        }
    
    @staticmethod
    def _as_2d_np(arr: np.ndarray, name: str) -> np.ndarray:
        """
        Function to ensure array dimension matches expected dimension
        
        :param arr: numpy array
        :param name: Name of array (training, validation etc.)
        """
        a= np.asarray(arr)
        if a.ndim == 1:
            return a.reshape(1, -1)
        if a.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D, got shape {a.shape}.")
        return a
    
    # -------------------------------- Public functions ------------------------------------------------
    def build_gp_model(self, x_train, y_train, device = 'cuda', 
                       noise_constraint : float = 1e-6, outputscale_constraint :float = 1e-6 ,
                       lengthscale_constraint : float = 1e-6):
        """
        Member function to Initialize GP Model before maximum likelihood estimation
        
        :param x_train: Training input data
        :param y_train: Training output data
        :param device: "cuda" or "cpu"
        :param noise_constraint: process noise covariance (small but not equal to zero, 1e-6 for example)
        :param outputscale_constraint: Must not be 0
        :param lengthscale_constraint: Must not be 0
        """
        if isinstance(x_train, np.ndarray):
            # Convert numpy array to torch tensors
            self.x_train = torch.from_numpy(x_train).to(device).float()
        elif torch.is_tensor(x_train):
            # Move to device
            self.x_train = x_train.to(device).float()
        else:
            raise ValueError(f"x_train must be np.ndarray or torch.Tensor, got {type(x_train)}")
        
        if isinstance(y_train, np.ndarray):     
            self.y_train = torch.from_numpy(y_train).to(device).float()
        elif torch.is_tensor(y_train):
            self.y_train = y_train.to(device).float()
        else:
            raise ValueError(f"y_train must be np.ndarray or torch.Tensor, got {type(y_train)}")
        
        # Define kernel based on config
        if self.config['kernel'].casefold() == "Matern12".casefold():
            base_kernel = MaternKernel(nu=0.5, ard_num_dims=self.x_train.shape[-1])
        elif self.config['kernel'].casefold() == "ReLU".casefold():
            base_kernel = InfiniteWidthBNNKernel(depth=self.config['depth'], device=device)
        else:
            raise ValueError(f"Kernel {self.config['kernel']} not implemented")

        # Model definition
        m = self.y_train.shape[-1]
        self.model = SingleTaskGP(
            train_X= self.x_train,
            train_Y = self.y_train,
            covar_module= ScaleKernel(base_kernel),
            outcome_transform=Standardize(m=m)
        ).to(device)

        # Hyperparameter constraints
        if self.config['kernel'].casefold() == "Matern12".casefold():
            # Noise lower bound
            self.model.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(noise_constraint))
            # Lengthscale lower bound
            self.model.covar_module.base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(lengthscale_constraint))
            # Outputscale lower bound
            self.model.covar_module.register_constraint("raw_outputscale", gpytorch.constraints.GreaterThan(outputscale_constraint))

        elif self.config['kernel'].casefold() == "ReLU".casefold():
            # Noise lower bound
            self.model.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(noise_constraint))
            # Outputscale lower bound
            self.model.covar_module.register_constraint("raw_outputscale", gpytorch.constraints.GreaterThan(outputscale_constraint))


        # Hyperparameter initialization
        self.init_hypers()
    
    def init_hypers(self):
        """
        Member function to initialize kernel hyperparameters to default and non-default values.
        
        """
        if self.config['version'] == 1:
            pass 

        elif self.config['version'] == 2:

            if self.config['kernel'].casefold() == "Matern12".casefold():
                self.model.covar_module.base_kernel.lengthscale = 0.3 
                self.model.covar_module.outputscale = 0.5  
                self.model.likelihood.noise = 1e-4 

            elif self.config['kernel'].casefold() == "ReLU".casefold():
                self.model.covar_module.base_kernel.weight_var = 0.5
                self.model.covar_module.base_kernel.bias_var = 1e-4
                self.model.covar_module.outputscale = 1
                self.model.likelihood.noise = 1e-2 
 
        elif self.config['version'] == 3:
            if self.config['kernel'].casefold() == "Matern12".casefold():
                self.model.covar_module.base_kernel.lengthscale = 3 
                self.model.covar_module.outputscale = 2  
                self.model.likelihood.noise = 1e0

            elif self.config['kernel'].casefold() == "ReLU".casefold():
                self.model.covar_module.base_kernel.weight_var = 2
                self.model.covar_module.base_kernel.bias_var = 1e-2
                self.model.covar_module.outputscale = 1
                self.model.likelihood.noise = 1e-2  
        else:
            print(f"Version {self.config['version']} not valid. Only version 1-3 supported")
            exit() 

    def train_model(self, savepath=f'{PROJECT_ROOT}\\reports\\Train_report.txt'):
        """        
        Member function that performs maximum likelihood estimation on the built GP Model. Has to be called after build_gp_model()

        :param savepath: Path to training report
        """
        if self.model is None:
            raise ValueError("Model is not built. Call build_gp_model(...) before train_model().")
        
        # Set model and likelihood to train mode
        self.model.train()
        self.model.likelihood.train()

        # Define optimizer
        mll =ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.05, max_iter=50,  line_search_fn="strong_wolfe")
        epochs = 10

        iters = {0 , 4, epochs-1}
        t_epoch = []
        iter_data = [] 
        iter_meta = {'timestamp_start' : datetime.now().isoformat(timespec="seconds")}

        print("Starting training loop...")
        t_tr0 = time.perf_counter()
        for i in range(epochs):
        
            def closure():
                optimizer.zero_grad(set_to_none=True)
                # Forward pass
                output = self.model(self.model.train_inputs[0])
                # Loss function
                loss = -mll(output, self.model.train_targets).sum() # Aggregate loss for all batches
                loss.backward()
                return loss

            t0 = time.perf_counter()
            optimizer.step(closure=closure)
            t_epoch.append(time.perf_counter() - t0)
           
            if i in iters:
                post_step_loss = self._eval_loss(mll)
                print(f'Post step loss at step {i+1}/{epochs} - {post_step_loss}')

                # Collect iteration data for training report
                hyp = self._get_hypers()
                iter_data.append({
                        'epoch' : i+1,
                        'loss_post' : post_step_loss,
                        'hypers' : hyp,
                        't_epoch_i' : t_epoch[-1]
                        })
                
        total_time = time.perf_counter() - t_tr0
        iter_meta['timestamp_end'] = datetime.now().isoformat(timespec='seconds')
        iter_meta['training_time'] = total_time
        iter_meta['epoch_times'] = t_epoch
        iter_meta['epochs'] = epochs

        self.generate_train_report(iter_data, iter_meta, savepath)

    def generate_train_report(self, iter_data, iter_metadata, savepath):
        """
        Member function to generate a training report. 
        
        """
        # General metadata
        metadata = {
            "kernel_type": self.config["kernel"],
            "version": self.config["version"],
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "Nsamp": int(self.x_train.shape[0]) if self.x_train is not None else None,
            "d": int(self.x_train.shape[1]) if self.x_train is not None else None,
            "m_latent": int(self.y_train.shape[1]) if self.y_train is not None else None,
        }
        
        if self.config["kernel"].casefold() == "ReLU".casefold():
            metadata["depth"] = int(self.config["depth"])
        
        epoch_times = np.asarray(iter_metadata['epoch_times'])
        
        if epoch_times.size > 0:
            t_min = float(epoch_times.min())
            t_med = float(np.median(epoch_times))
            t_max = float(epoch_times.max())
        else:
            t_min = t_med = t_max = float("nan")

        report_metadata = {**metadata, **iter_metadata}
        lines = []
        lines.append('============================GP TRAINING REPORT========================')
        for k in ["kernel_type", "version", "depth", "device", "dtype", "epochs", "Nsamp", "d", "m_latent",
                  "timestamp_start", "timestamp_end", "training_time"]:
            if k in report_metadata:
                lines.append(f'{k} : {report_metadata[k]}')

        lines.append("")
        lines.append("---- Epoch timing ----")
        lines.append(f"min/median/max per step (s): {t_min:.6g} / {t_med:.6g} / {t_max:.6g}")
        lines.append("")
        
        for data in iter_data:
            lines.append(f"------Step {data['epoch']}---------")
            lines.append(f"Loss value : {data['loss_post']}")
            lines.append(f"Time taken for step {data['epoch']} - {data['t_epoch_i']}")
            hypers = data.get("hypers",{})
            lines.append("Hyperparameters:")
            for name, stats in hypers.items():
                if isinstance(stats,dict) and ("val" in stats):
                    lines.append(f"{name} : {stats['val']:.6g}")
                else: 
                    lines.append(
                        f"{name} : min={stats['min']:.6g}, median={stats['median']:.6g}, "
                        f"max={stats['max']:.6g}"
                    )
        lines.append("")
        self.train_report = "\n".join(lines)

        with open(savepath, 'w', encoding='utf-8') as f:
            f.write(self.train_report)

    def infer(self, x_test_np, is_scaled = True):
        """
        Member function to infer for a single new validation point p.
        
        :param x_test_np: New test point
        :param is_scaled: Scaled input flag. If input is not scaled from [0,1], set to False.
        """
        if self.model is None:
            raise ValueError("Load model using load_model")
        if (not is_scaled) and (self.prep is None):
            raise ValueError("is_scaled=False requires fitted PreProcessor")

        x_np = self._as_2d_np(x_test_np, "x_test_np")

        if not is_scaled:
            x_np = self.prep.x_scaler.transform(x_np)
        
        # Torch on same device/dtype as model
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        x_t = torch.from_numpy(x_np).to(device=device, dtype=dtype)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self.model.posterior(x_t)
            # Mean in latent space
            y_pca_mean = post.mean
            # Variance in latent space
            y_pca_var = post.variance
        
        y_pca_mean_np = y_pca_mean.detach().cpu().numpy()
        y_pca_var_np = y_pca_var.detach().cpu().numpy()

        # Map latent space -> full output space using prep object
        y_mean_scaled = self.prep.reverse_pca_Y(y_pca_mean_np)
        y_mean = self.prep.reverse_scale_Y(y_mean_scaled)

        # ToDo: Variance to output space conversion

        return y_mean

    def analyse_model(self, x_train, x_test_np, y_train, y_test, path: str | None, generate_report: bool):
        """
        Docstring for analyse_model
        
        Generate validation(or test) report for training and testing data.
        """
        res_train = self.generate_test_report(x_test_np=x_train, y_test=y_train, split='Training Data')
        res_test = self.generate_test_report(x_test_np=x_test_np, y_test=y_test, split='Testing Data')

        if generate_report:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(res_train['report'])
                f.write("\n\n")
                f.write(res_test['report'])
                f.write("\n")
            
        return res_train, res_test


    def generate_test_report(self, x_test_np, y_test, split):
        """
        Member function to generate test (or validation) report.

        """
        # Convert numpy test data to tensors
        x_test_torch = torch.from_numpy(x_test_np).to(self.device).float()
        
        # Inference
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self.model.posterior(x_test_torch)
            y_hat_pca = post.mean 
        
        # Inverse PCA and scaling to obtain output in required space
        y_hat_pca_np = y_hat_pca.detach().cpu().numpy()
        y_hat_scaled = self.prep.reverse_pca_Y(y_hat_pca_np)
        y_hat = self.prep.reverse_scale_Y(y_hat_scaled)

        # Inverse y transforms if training data
        if split.casefold() == 'Training Data'.casefold():
            y_scaled = self.prep.reverse_pca_Y(y_test)
            y = self.prep.reverse_scale_Y(y_scaled)
        else:
            y = y_test

        assert y_hat.shape[-1] == y.shape[-1]

        # L_inf error calculation
        AbsE = np.abs(y_hat - y)

        e_sample = AbsE.max(axis=1) # L_inf error for each sample
        e_dim = AbsE.max(axis=0) # L_inf error for each dimension

        flat_idx = AbsE.reshape(-1).argmax()
        i_star = flat_idx // AbsE.shape[1]
        j_star = flat_idx % AbsE.shape[1] 

        err_var = np.var(e_sample)
        qs = np.quantile(e_sample, [0.0, 0.5, 0.9, 0.99, 1.0])
        lines=[]
        lines.append(f"========================================Error on {split}================================================")
        lines.append(f"Worst error at sample {i_star}, dim {j_star}. "
                     f"|err|={AbsE[i_star, j_star]:.6g}," 
                     f"true output={y[i_star, j_star]:.6g}, predicted output = {y_hat[i_star, j_star]:.6g}")
        
        lines.append(f"Per-sample L_inf quantiles: min={qs[0]:.6g}, median={qs[1]:.6g}, p90={qs[2]:.6g}, p99={qs[3]:.6g}, max={qs[4]:.6g}")
        lines.append(f"Per-sample L_inf Error Variance : {err_var}")
        lines.append(f"Per-sample Mean L_inf Error : {np.mean(e_sample)}")
        top_idx = np.argsort(-e_dim)[:10]
        lines.append(f"Top-{10} worst dual coordinates (dim : worst_abs_err):")
        for j in top_idx:
            lines.append(f"  {j:2d}: {e_dim[j]:.6g}")
        
        report = "\n".join(lines)

        e_dict = {'e_sample':e_sample, 'e_dim':e_dim, 'emax_idx':(int(i_star), int(j_star)), 'pred_output':y_hat, 'actual_output':y}
        e_dict["report"] = report

        return e_dict 

    def save_model(self, path):
        """
        Member function to save trained GP models
        
        :param path: Save location
        """
        if self.model is None or self.x_train is None or self.y_train is None:
            raise ValueError("Nothing to save. Build and train the model first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)  

        # Save model state dict and training variables
        save_dict = {
            'model_state_dict' : self.model.state_dict(),
            'train_X' : self.x_train,
            'train_Y' : self.y_train,
            'config' : self.config
        } 

        torch.save(save_dict, path)

    def load_model(self, path):
        """
        Member function to load saved GP models
        
        :param path: Load path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)  

        ckpt = torch.load(path, map_location="cpu")
        self.build_gp_model(ckpt['train_X'], ckpt['train_Y'], device=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])

        # set eval mode
        self.model.eval()
        self.model.likelihood.eval()

        # cache warmup 
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _ = self.model.posterior(self.x_train[:1])


