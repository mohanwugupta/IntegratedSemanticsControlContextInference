import torch
import torch.nn as nn
import torch.nn.functional as F


class CICOModel(nn.Module):
    def __init__(
        self,
        input_d: int = 293,
        embedding_d: int = 64,
        context_d: int = 128,
        context_dependent_d: int = 128,
        task_output_d: int = 385,
        nonlinearity: nn.Module = nn.ReLU,
        lr=0.001,
        task_loss_wt=0.5,
        device="cpu",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        input_d : int
            The dimensionality of the input.
        embedding_d : int
            The dimensionality of the embedding (context independent) layer.
        context_d : int
            The dimensionality of the context layer.
        context_dependent_d : int
            The dimensionality of the context dependent layer.
        task_output_d : int
            The dimensionality of the context prediction output.
        nonlinearity : nn.Module, optional
            The nonlinearity to use. Defaults to nn.ReLU.
        lr : float, optional
            The learning rate. Defaults to .001.
        task_loss_wt : float, optional
            The weight of the context prediction loss. Defaults to .5.
        """
        super().__init__()

        self.input_to_independent = nn.Embedding(input_d, embedding_d)
        self.independent_to_context = nn.Linear(embedding_d, context_d)
        self.context_to_dependent = nn.Linear(context_d, context_dependent_d)
        self.independent_to_dependent = nn.Linear(embedding_d, context_dependent_d)
        self.dependent_to_ft_output = nn.Linear(context_dependent_d, 1, bias=False)
        self.context_to_task_output = nn.Linear(context_d, task_output_d)

        self.nonlinearity = nonlinearity()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.epoch = 0
        self.task_loss_wt = task_loss_wt
        self.metrics = []
        self.device = device

    def get_independent_rep(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_to_independent(x)

    def get_context_rep(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.get_independent_rep(x)
        average_embedding = embedding.mean(dim=1)
        return self.nonlinearity(self.independent_to_context(average_embedding))

    def get_context_dependent_rep(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if len(context.shape) == 2:
            context = context.unsqueeze(1).repeat(1, x.shape[1], 1)
        return self.nonlinearity(
            self.independent_to_dependent(self.get_independent_rep(x))
            + self.context_to_dependent(context)
        )

    def get_ft_output(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        context_dependent = self.get_context_dependent_rep(x, context)
        return self.dependent_to_ft_output(context_dependent)

    def train_on_batch(
        self,
        support_x: torch.tensor,
        support_y: torch.tensor,
        support_c: torch.tensor,
        query_x: torch.tensor,
        query_y: torch.tensor,
        query_c: torch.tensor,
    ) -> tuple:
        context_rep = self.get_context_rep(support_x)
        context_pred = self.context_to_task_output(context_rep)
        support_ft_pred = self.get_ft_output(support_x, context_rep).squeeze(-1)
        query_ft_pred = self.get_ft_output(query_x, context_rep).squeeze(-1)
        c_loss = F.cross_entropy(context_pred, support_c[:, 0].long())
        y_loss = (
            F.binary_cross_entropy_with_logits(support_ft_pred, support_y.float())
            * support_y.shape[1]
        )
        y_loss += (
            F.binary_cross_entropy_with_logits(query_ft_pred, query_y.float())
            * query_y.shape[1]
        )
        y_loss /= support_y.shape[1] + query_y.shape[1]
        return y_loss, c_loss

    def train_on_batch_matrix(
        self,
        support_x: torch.tensor,
        query_x: torch.tensor,
        c: torch.tensor,
        y: torch.tensor,
    ) -> tuple:
        context_rep = self.get_context_rep(support_x)
        context_pred = self.context_to_task_output(context_rep)
        ft_pred = self.get_ft_output(query_x, context_rep).squeeze(-1)
        c_loss = F.cross_entropy(context_pred, c)
        y_loss = F.binary_cross_entropy_with_logits(ft_pred, y)
        return y_loss, c_loss

    def fit(
        self,
        train_loader,
        n_epochs,
    ):
        for _ in range(n_epochs):
            for batch_idx, batch in enumerate(train_loader):
                context_rep = self.get_context_rep(batch["support_x"].to(self.device))
                task_pred = self.context_to_task_output(context_rep)
                support_ft_pred = self.get_ft_output(
                    batch["support_x"].to(self.device), context_rep
                ).squeeze(-1)
                query_ft_pred = self.get_ft_output(
                    batch["query_x"].to(self.device), context_rep
                ).squeeze(-1)

                task_loss = F.cross_entropy(
                    task_pred, batch["support_c"][:, 0].to(self.device)
                )
                support_ft_loss = F.binary_cross_entropy_with_logits(
                    support_ft_pred, batch["support_y"][:, :].to(self.device)
                )
                query_ft_loss = F.binary_cross_entropy_with_logits(
                    query_ft_pred, batch["query_y"][:, :].to(self.device)
                )
                ft_loss = support_ft_loss + query_ft_loss
                loss = self.task_loss_wt * task_loss + (1 - self.task_loss_wt) * ft_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                task_acc = (
                    (task_pred.argmax(-1).cpu() == batch["support_c"][:, 0].cpu())
                    .float()
                    .mean()
                    .item()
                )
                support_ft_acc = (
                    ((support_ft_pred > 0).cpu() == batch["support_y"][:, :].cpu())
                    .float()
                    .mean()
                    .item()
                )
                query_ft_acc = (
                    ((query_ft_pred > 0).cpu() == batch["query_y"][:, :].cpu())
                    .float()
                    .mean()
                    .item()
                )
                self.metrics.append(
                    {
                        "task_acc": task_acc,
                        "support_ft_acc": support_ft_acc,
                        "query_ft_acc": query_ft_acc,
                        "task_loss": task_loss.item(),
                        "support_ft_loss": support_ft_loss.item(),
                        "query_ft_loss": query_ft_loss.item(),
                        "loss": loss.item(),
                        "epoch": self.epoch,
                        "batch_idx": batch_idx,
                    }
                )
            self.epoch += 1
